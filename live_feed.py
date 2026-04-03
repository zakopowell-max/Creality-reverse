#!/usr/bin/env python3
"""
Live depth feed from Creality CR-Scan Otter — GTK3Agg display (Wayland-safe).
Press Q or close the window to quit.

Usage: python3 live_feed.py [--max-depth METRES]
"""
import v4l2, fcntl, mmap, select, time, argparse, sys
import numpy as np
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt

DEPTH_DEV = '/dev/video2'
DEPTH_W, DEPTH_H = 640, 400
DEPTH_SCALE = 0.005   # m/unit, calibrated 2026-04-03
NBUF = 4


# ── V4L2 helpers ─────────────────────────────────────────────────────────────

def _set_fmt(fd):
    fmt = v4l2.v4l2_format()
    fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    fmt.fmt.pix.width = DEPTH_W
    fmt.fmt.pix.height = DEPTH_H
    fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_Y10
    fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
    fcntl.ioctl(fd, v4l2.VIDIOC_S_FMT, fmt)

def _alloc_buffers(fd):
    req = v4l2.v4l2_requestbuffers()
    req.count = NBUF
    req.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    req.memory = v4l2.V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, v4l2.VIDIOC_REQBUFS, req)
    bufs = []
    for i in range(req.count):
        b = v4l2.v4l2_buffer()
        b.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        b.memory = v4l2.V4L2_MEMORY_MMAP
        b.index = i
        fcntl.ioctl(fd, v4l2.VIDIOC_QUERYBUF, b)
        mm = mmap.mmap(fd.fileno(), b.length, offset=b.m.offset)
        bufs.append(mm)
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, b)
    return bufs

def _free_buffers(fd, bufs):
    for mm in bufs:
        mm.close()
    req = v4l2.v4l2_requestbuffers()
    req.count = 0
    req.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    req.memory = v4l2.V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, v4l2.VIDIOC_REQBUFS, req)

def _streamon(fd):
    fcntl.ioctl(fd, v4l2.VIDIOC_STREAMON,
                v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))

def _streamoff(fd):
    fcntl.ioctl(fd, v4l2.VIDIOC_STREAMOFF,
                v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))


def prime_device():
    print("Priming scanner...", end=' ', flush=True)
    fd = open(DEPTH_DEV, 'rb+', buffering=0)
    _set_fmt(fd)
    bufs = _alloc_buffers(fd)
    _streamon(fd)
    for _ in range(30):
        r, _, _ = select.select([fd], [], [], 1.0)
        if not r:
            break
        b = v4l2.v4l2_buffer()
        b.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        b.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, b)
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, b)
        if b.bytesused >= 320000:
            break
    _streamoff(fd)
    _free_buffers(fd, bufs)
    fd.close()
    print("done")


def open_stream():
    fd = open(DEPTH_DEV, 'rb+', buffering=0)
    _set_fmt(fd)
    bufs = _alloc_buffers(fd)
    _streamon(fd)
    return fd, bufs


def close_stream(fd, bufs):
    _streamoff(fd)
    _free_buffers(fd, bufs)
    fd.close()


def decode_y10(raw, w, h):
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
    n = len(arr) // 5
    b = arr[:n*5].reshape(n, 5)
    px = np.zeros(n * 4, dtype=np.uint16)
    px[0::4] = (b[:,0] << 2) | (b[:,1] >> 6)
    px[1::4] = ((b[:,1] & 0x3f) << 4) | (b[:,2] >> 4)
    px[2::4] = ((b[:,2] & 0x0f) << 6) | (b[:,3] >> 2)
    px[3::4] = ((b[:,3] & 0x03) << 8) | b[:,4]
    return px[:w * h].reshape(h, w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-depth', type=float, default=2.0,
                        help='Max depth for colour scale (metres, default 2.0)')
    args = parser.parse_args()
    max_raw = int(args.max_depth / DEPTH_SCALE)

    prime_device()
    fd, bufs = open_stream()

    # Skip warm-up empty frames
    skipped = 0
    while True:
        r, _, _ = select.select([fd], [], [], 5.0)
        if not r:
            print("ERROR: no frames from device")
            close_stream(fd, bufs)
            sys.exit(1)
        b = v4l2.v4l2_buffer()
        b.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        b.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, b)
        if b.bytesused >= 320000:
            fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, b)
            break
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, b)
        skipped += 1
    print(f"Skipped {skipped} warm-up frames. Live feed starting — close window or Ctrl-C to quit.")

    # Set up matplotlib figure
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6.25))
    fig.patch.set_facecolor('#111')
    ax.set_facecolor('#111')
    ax.axis('off')

    # Blank initial frame
    display_img = np.zeros((DEPTH_H, DEPTH_W), dtype=np.float32)
    im = ax.imshow(display_img, cmap='plasma', vmin=0, vmax=max_raw,
                   interpolation='nearest', aspect='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label('Depth (m)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    # Colourbar ticks in metres
    tick_vals = np.linspace(0, max_raw, 6)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f'{v*DEPTH_SCALE:.1f}m' for v in tick_vals])

    title = ax.set_title('Depth feed — starting...', color='white', fontsize=11)
    plt.tight_layout(pad=0.5)
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Blit background for speed
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    frame_count = 0
    fps_t = time.time()
    running = True

    def on_close(_):
        nonlocal running
        running = False

    fig.canvas.mpl_connect('close_event', on_close)

    try:
        while running:
            r, _, _ = select.select([fd], [], [], 0.1)
            if not r:
                fig.canvas.flush_events()
                continue

            b = v4l2.v4l2_buffer()
            b.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
            b.memory = v4l2.V4L2_MEMORY_MMAP
            fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, b)

            if b.bytesused < 320000:
                fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, b)
                fig.canvas.flush_events()
                continue

            raw = bytes(bufs[b.index][:320000])
            fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, b)

            depth = decode_y10(raw, DEPTH_W, DEPTH_H).astype(np.float32)
            # Zero out pixels above max range (treat as no-return)
            depth[depth > max_raw] = 0
            frame_count += 1

            # Update display via blitting
            im.set_data(depth)
            fig.canvas.restore_region(bg)
            ax.draw_artist(im)

            now = time.time()
            if now - fps_t >= 1.5:
                fps = frame_count / (now - fps_t)
                frame_count = 0
                fps_t = now
                cy = slice(DEPTH_H//2-10, DEPTH_H//2+10)
                cx = slice(DEPTH_W//2-16, DEPTH_W//2+16)
                center_raw = depth[cy, cx]
                cnz = center_raw[center_raw > 0]
                if len(cnz):
                    cdist = f"{cnz.mean()*DEPTH_SCALE*100:.0f}cm"
                else:
                    cdist = "no-return"
                title.set_text(f'{fps:.1f} fps  |  centre {cdist}  |  close window to quit')
                ax.draw_artist(title)

            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        pass
    finally:
        close_stream(fd, bufs)
        plt.close(fig)
        print("Stream closed.")


if __name__ == '__main__':
    main()
