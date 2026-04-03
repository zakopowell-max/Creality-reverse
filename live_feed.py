#!/usr/bin/env python3
"""
Live depth feed from Creality CR-Scan Otter — GTK3Agg display (Wayland-safe).

Usage:
  python3 live_feed.py              # 2D depth image only (fastest)
  python3 live_feed.py --3d         # depth image + live 3D point cloud
  python3 live_feed.py --snapshot   # capture one PLY then view it via Open3D web
  python3 live_feed.py --max-depth 1.5

Press Q / close window to quit.
"""
import v4l2, fcntl, mmap, select, time, argparse, sys
import numpy as np
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3d projection

DEPTH_DEV = '/dev/video2'
DEPTH_W, DEPTH_H = 640, 400
DEPTH_SCALE = 0.005   # m/unit, calibrated 2026-04-03
DEPTH_FX, DEPTH_FY = 620.0, 620.0
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


# ── Device lifecycle ─────────────────────────────────────────────────────────

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


# ── Data processing ──────────────────────────────────────────────────────────

def decode_y10(raw, w=DEPTH_W, h=DEPTH_H):
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
    n = len(arr) // 5
    b = arr[:n*5].reshape(n, 5)
    px = np.zeros(n * 4, dtype=np.uint16)
    px[0::4] = (b[:,0] << 2) | (b[:,1] >> 6)
    px[1::4] = ((b[:,1] & 0x3f) << 4) | (b[:,2] >> 4)
    px[2::4] = ((b[:,2] & 0x0f) << 6) | (b[:,3] >> 2)
    px[3::4] = ((b[:,3] & 0x03) << 8) | b[:,4]
    return px[:w * h].reshape(h, w)


def depth_to_points(depth_img, min_m=0.05, max_m=5.0, max_pts=6000):
    cx, cy = DEPTH_W / 2, DEPTH_H / 2
    h, w = depth_img.shape
    ys, xs = np.mgrid[0:h, 0:w]
    Z = depth_img.astype(np.float32) * DEPTH_SCALE
    valid = (Z > min_m) & (Z < max_m)
    X = (xs[valid] - cx) * Z[valid] / DEPTH_FX
    Y = (ys[valid] - cy) * Z[valid] / DEPTH_FY
    pts = np.column_stack([X, Y, Z[valid]])
    # Subsample for display speed
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
    return pts


# ── Snapshot mode ────────────────────────────────────────────────────────────

def do_snapshot(max_m):
    """Capture one clean frame, save PLY, view via Open3D draw_plotly."""
    from capture_and_cloud import (capture_frames, decode_y10_packed,
                                   depth_to_pointcloud, save_ply)
    import open3d as o3d

    fd, bufs = open_stream()
    raw_frames = capture_frames(fd, bufs, 5)
    close_stream(fd, bufs)

    depth_sum = sum(decode_y10_packed(r, DEPTH_W, DEPTH_H).astype(np.float32)
                    for r in raw_frames) / len(raw_frames)
    pts = depth_to_pointcloud(depth_sum, DEPTH_FX, DEPTH_FY,
                              DEPTH_W/2, DEPTH_H/2, DEPTH_SCALE,
                              min_depth=0.05, max_depth=max_m)
    ply_path = '/tmp/snapshot.ply'
    save_ply(pts, ply_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    z = pts[:, 2]
    t = np.clip((z - 0.05) / (max_m - 0.05), 0, 1)
    import matplotlib.cm as cmx
    colours = cmx.plasma(t)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colours)
    print(f"Saved {len(pts)} points to {ply_path}")
    print("Opening in browser (close tab when done)...")
    o3d.visualization.draw_plotly([pcd])


# ── Main live loop ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--3d', action='store_true', dest='show3d',
                        help='Show live 3D point cloud alongside depth image')
    parser.add_argument('--snapshot', action='store_true',
                        help='Capture one frame, save PLY, view in browser')
    parser.add_argument('--max-depth', type=float, default=2.0,
                        help='Max depth for colour scale (metres, default 2.0)')
    args = parser.parse_args()
    max_raw = int(args.max_depth / DEPTH_SCALE)

    prime_device()

    if args.snapshot:
        do_snapshot(args.max_depth)
        return

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
    print(f"Skipped {skipped} warm-up frames. Live feed — close window or Ctrl-C to quit.")

    # ── Figure layout ────────────────────────────────────────────────────────
    plt.ion()
    if args.show3d:
        fig = plt.figure(figsize=(14, 5), facecolor='#111')
        ax2d = fig.add_subplot(1, 2, 1)
        ax3d = fig.add_subplot(1, 2, 2, projection='3d')
        ax3d.set_facecolor('#111')
        ax3d.set_xlabel('X', color='#888', fontsize=8)
        ax3d.set_ylabel('Z (depth)', color='#888', fontsize=8)
        ax3d.set_zlabel('Y', color='#888', fontsize=8)
        ax3d.tick_params(colors='#555', labelsize=7)
        for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor('#333')
        scat3d = None
    else:
        fig, ax2d = plt.subplots(figsize=(10, 6.25))
        fig.patch.set_facecolor('#111')

    ax2d.set_facecolor('#111')
    ax2d.axis('off')

    blank = np.zeros((DEPTH_H, DEPTH_W), dtype=np.float32)
    im = ax2d.imshow(blank, cmap='plasma', vmin=0, vmax=max_raw,
                     interpolation='nearest', aspect='auto')
    cbar = fig.colorbar(im, ax=ax2d, fraction=0.025, pad=0.01)
    cbar.set_label('Depth (m)', color='white')
    tick_vals = np.linspace(0, max_raw, 6)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f'{v*DEPTH_SCALE:.1f}m' for v in tick_vals])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    title = ax2d.set_title('Starting...', color='white', fontsize=10)
    plt.tight_layout(pad=0.5)
    fig.canvas.draw()
    fig.canvas.flush_events()

    running = [True]
    fig.canvas.mpl_connect('close_event', lambda _: running.__setitem__(0, False))

    # For 3D: only redraw every N depth frames (matplotlib 3D is slow)
    cloud_every = 5
    cloud_counter = 0
    frame_count = 0
    fps_t = time.time()

    try:
        while running[0]:
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

            depth = decode_y10(raw).astype(np.float32)
            depth[depth > max_raw] = 0
            frame_count += 1
            cloud_counter += 1

            # ── 2D depth update ──────────────────────────────────────────────
            im.set_data(depth)

            now = time.time()
            if now - fps_t >= 1.5:
                fps = frame_count / (now - fps_t)
                frame_count = 0
                fps_t = now
                cy = slice(DEPTH_H//2-10, DEPTH_H//2+10)
                cx = slice(DEPTH_W//2-16, DEPTH_W//2+16)
                cnz = depth[cy, cx]
                cnz = cnz[cnz > 0]
                cdist = f"{cnz.mean()*DEPTH_SCALE*100:.0f}cm" if len(cnz) else "no-return"
                mode = " (+3D)" if args.show3d else ""
                title.set_text(f'{fps:.1f} fps{mode}  |  centre {cdist}  |  close to quit')

            # ── 3D point cloud update (every N frames) ───────────────────────
            if args.show3d and cloud_counter >= cloud_every:
                cloud_counter = 0
                pts = depth_to_points(depth, max_m=args.max_depth)
                if len(pts):
                    if scat3d is not None:
                        scat3d.remove()
                    z = pts[:, 2]
                    t = np.clip((z - 0.05) / (args.max_depth - 0.05), 0, 1)
                    import matplotlib.cm as cmx
                    colours = cmx.plasma(t)
                    # matplotlib 3D: X=left/right, Y=depth, Z=up/down
                    scat3d = ax3d.scatter(pts[:,0], pts[:,2], -pts[:,1],
                                         c=colours, s=0.5, alpha=0.6,
                                         depthshade=False)
                    ax3d.set_xlim(-1, 1)
                    ax3d.set_ylim(0, args.max_depth)
                    ax3d.set_zlim(-0.5, 0.5)
                fig.canvas.draw()
            else:
                # Fast blit for 2D-only frames
                fig.canvas.draw_idle()

            fig.canvas.flush_events()

    except KeyboardInterrupt:
        pass
    finally:
        close_stream(fd, bufs)
        plt.close(fig)
        print("Stream closed.")


if __name__ == '__main__':
    main()
