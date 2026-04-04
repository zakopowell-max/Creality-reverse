#!/usr/bin/env python3
"""
Live depth feed from Creality CR-Scan Otter — GTK3Agg display (Wayland-safe).

Usage:
  python3 live_feed.py              # 2D depth image only (fastest)
  python3 live_feed.py --3d         # depth image + live 3D point cloud
  python3 live_feed.py --ir         # depth image + IR stream side by side
  python3 live_feed.py --colour     # depth + colour + RGB point cloud
  python3 live_feed.py --snapshot   # capture one PLY then view it via Open3D web
  python3 live_feed.py --max-depth 1.5

Note: --colour uses rough pixel-aligned mapping (same x,y for depth+colour).
Proper extrinsic calibration is TODO — good enough to see Rubik's cube colours.

Press Q / close window to quit.
"""
import v4l2, fcntl, mmap, select, time, argparse, sys
import numpy as np
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3d projection
from devices import find_otter_devices

DEPTH_DEV, IR_DEV, COLOUR_DEV = find_otter_devices()

DEPTH_W, DEPTH_H = 640, 400
DEPTH_SCALE = 0.005   # m/unit, calibrated 2026-04-03
DEPTH_FX, DEPTH_FY = 620.0, 620.0

IR_W, IR_H = 1280, 800
IR_FRAME_SIZE = IR_W * IR_H * 10 // 8  # 1280000 bytes


COLOUR_W, COLOUR_H = 640, 480
COLOUR_FRAME_SIZE = COLOUR_W * COLOUR_H * 2  # YUYV: 2 bytes/pixel
# Colour is 640x480, depth is 640x400 — crop 40px top+bottom to align vertically
COLOUR_CROP_Y = (COLOUR_H - DEPTH_H) // 2   # = 40

NBUF = 4


# ── V4L2 helpers ─────────────────────────────────────────────────────────────

def _set_fmt(fd, w, h):
    fmt = v4l2.v4l2_format()
    fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    fmt.fmt.pix.width = w
    fmt.fmt.pix.height = h
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

def _prime(device, w, h, min_frame_size, pixfmt=None):
    fd = open(device, 'rb+', buffering=0)
    if pixfmt is not None:
        fmt = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = w; fmt.fmt.pix.height = h
        fmt.fmt.pix.pixelformat = pixfmt
        fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
        fcntl.ioctl(fd, v4l2.VIDIOC_S_FMT, fmt)
    else:
        _set_fmt(fd, w, h)
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
        if b.bytesused >= min_frame_size:
            break
    _streamoff(fd)
    _free_buffers(fd, bufs)
    fd.close()


def prime_device(with_ir=False, with_colour=False):
    print("Priming depth...", end=' ', flush=True)
    _prime(DEPTH_DEV, DEPTH_W, DEPTH_H, DEPTH_W * DEPTH_H * 10 // 8)
    print("done", end='')
    if with_ir:
        print(", IR...", end=' ', flush=True)
        _prime(IR_DEV, IR_W, IR_H, IR_FRAME_SIZE)
        print("done", end='')
    if with_colour:
        print(", colour...", end=' ', flush=True)
        _prime(COLOUR_DEV, COLOUR_W, COLOUR_H, COLOUR_FRAME_SIZE, pixfmt=v4l2.V4L2_PIX_FMT_YUYV)
        print("done", end='')
    print()


def open_stream(device, w, h):
    fd = open(device, 'rb+', buffering=0)
    _set_fmt(fd, w, h)
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


def decode_yuyv(raw, w=COLOUR_W, h=COLOUR_H):
    """YUYV packed → RGB uint8 (h, w, 3). Each 4 bytes = 2 pixels: Y0 U Y1 V."""
    data = np.frombuffer(raw, dtype=np.uint8).reshape(h, w // 2, 4).astype(np.float32)
    y = np.empty((h, w), dtype=np.float32)
    y[:, 0::2] = data[:, :, 0]
    y[:, 1::2] = data[:, :, 2]
    u = np.repeat(data[:, :, 1], 2, axis=1) - 128.0
    v = np.repeat(data[:, :, 3], 2, axis=1) - 128.0
    r = np.clip(y + 1.402 * v,                      0, 255)
    g = np.clip(y - 0.344136 * u - 0.714136 * v,    0, 255)
    b = np.clip(y + 1.772 * u,                      0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def open_colour_stream():
    """Open colour stream with boosted gain for indoor/dim environments."""
    import subprocess
    subprocess.run(['v4l2-ctl', '-d', COLOUR_DEV,
                    '-c', 'gain=255', '-c', 'brightness=200'],
                   capture_output=True)
    fd = open(COLOUR_DEV, 'rb+', buffering=0)
    fmt = v4l2.v4l2_format()
    fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    fmt.fmt.pix.width = COLOUR_W
    fmt.fmt.pix.height = COLOUR_H
    fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_YUYV
    fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
    fcntl.ioctl(fd, v4l2.VIDIOC_S_FMT, fmt)
    bufs = _alloc_buffers(fd)
    _streamon(fd)
    return fd, bufs


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

    fd, bufs = open_stream(DEPTH_DEV, DEPTH_W, DEPTH_H)
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
    parser.add_argument('--ir', action='store_true',
                        help='Show IR stream alongside depth image')
    parser.add_argument('--colour', action='store_true',
                        help='Show colour camera + RGB-coloured point cloud')
    parser.add_argument('--snapshot', action='store_true',
                        help='Capture one frame, save PLY, view in browser')
    parser.add_argument('--max-depth', type=float, default=2.0,
                        help='Max depth for colour scale (metres, default 2.0)')
    args = parser.parse_args()
    max_raw = int(args.max_depth / DEPTH_SCALE)

    prime_device(with_ir=args.ir, with_colour=args.colour)

    if args.snapshot:
        do_snapshot(args.max_depth)
        return

    depth_fd, depth_bufs = open_stream(DEPTH_DEV, DEPTH_W, DEPTH_H)
    ir_fd,    ir_bufs    = (open_stream(IR_DEV, IR_W, IR_H) if args.ir else (None, None))
    col_fd,   col_bufs   = (open_colour_stream() if args.colour else (None, None))

    # Skip warm-up empty frames on depth stream
    skipped = 0
    while True:
        r, _, _ = select.select([depth_fd], [], [], 5.0)
        if not r:
            print("ERROR: no frames from device")
            close_stream(depth_fd, depth_bufs)
            sys.exit(1)
        b = v4l2.v4l2_buffer()
        b.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        b.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(depth_fd, v4l2.VIDIOC_DQBUF, b)
        if b.bytesused >= DEPTH_W * DEPTH_H * 10 // 8:
            fcntl.ioctl(depth_fd, v4l2.VIDIOC_QBUF, b)
            break
        fcntl.ioctl(depth_fd, v4l2.VIDIOC_QBUF, b)
        skipped += 1
    mode_str = " (+IR)" if args.ir else (" (+colour)" if args.colour else (" (+3D)" if args.show3d else ""))
    print(f"Skipped {skipped} warm-up frames. Live feed{mode_str} — close window or Ctrl-C to quit.")

    # ── Figure layout ────────────────────────────────────────────────────────
    plt.ion()
    n_panels = 1 + bool(args.show3d) + bool(args.ir) + bool(args.colour)
    fig = plt.figure(figsize=(10 * n_panels // 1, 5) if n_panels > 1 else (10, 6.25),
                     facecolor='#111')

    if n_panels == 1:
        ax2d = fig.add_subplot(1, 1, 1)
    else:
        ax2d = fig.add_subplot(1, n_panels, 1)

    ax2d.set_facecolor('#111')
    ax2d.axis('off')

    blank_d = np.zeros((DEPTH_H, DEPTH_W), dtype=np.float32)
    im = ax2d.imshow(blank_d, cmap='plasma', vmin=0, vmax=max_raw,
                     interpolation='nearest', aspect='auto')
    cbar = fig.colorbar(im, ax=ax2d, fraction=0.025, pad=0.01)
    cbar.set_label('Depth (m)', color='white')
    tick_vals = np.linspace(0, max_raw, 6)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f'{v*DEPTH_SCALE:.1f}m' for v in tick_vals])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    ax2d.set_title('Depth', color='white', fontsize=9)

    # IR panel
    ax_ir = im_ir = None
    if args.ir:
        ax_ir = fig.add_subplot(1, n_panels, 2)
        ax_ir.set_facecolor('#111')
        ax_ir.axis('off')
        ax_ir.set_title('IR  (1280×800)', color='white', fontsize=9)
        blank_ir = np.zeros((IR_H, IR_W), dtype=np.float32)
        im_ir = ax_ir.imshow(blank_ir, cmap='gray', vmin=0, vmax=1023,
                             interpolation='nearest', aspect='auto')

    # Colour panel
    ax_col = im_col = None
    latest_colour = [None]   # shared between capture and 3D scatter
    if args.colour:
        panel_idx = 2 + bool(args.ir)
        ax_col = fig.add_subplot(1, n_panels, panel_idx)
        ax_col.set_facecolor('#111')
        ax_col.axis('off')
        ax_col.set_title('Colour  (rough align)', color='white', fontsize=9)
        blank_col = np.zeros((DEPTH_H, COLOUR_W, 3), dtype=np.uint8)
        im_col = ax_col.imshow(blank_col, interpolation='nearest', aspect='auto')

    # 3D panel (always last)
    ax3d = scat3d = None
    if args.show3d:
        ax3d = fig.add_subplot(1, n_panels, n_panels, projection='3d')
        ax3d.set_facecolor('#111')
        ax3d.set_xlabel('X', color='#888', fontsize=8)
        ax3d.set_ylabel('Z', color='#888', fontsize=8)
        ax3d.set_zlabel('Y', color='#888', fontsize=8)
        ax3d.tick_params(colors='#555', labelsize=7)
        for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor('#333')

    title = fig.suptitle('Starting...', color='white', fontsize=10)
    plt.tight_layout(pad=0.5)
    fig.canvas.draw()
    fig.canvas.flush_events()

    running = [True]
    fig.canvas.mpl_connect('close_event', lambda _: running.__setitem__(0, False))

    cloud_every = 5
    cloud_counter = 0
    frame_count = 0
    fps_t = time.time()

    try:
        while running[0]:
            # Poll all active streams
            fds = [depth_fd] + ([ir_fd] if ir_fd else []) + ([col_fd] if col_fd else [])
            r, _, _ = select.select(fds, [], [], 0.05)
            if not r:
                fig.canvas.flush_events()
                continue

            # ── Depth frame ──────────────────────────────────────────────────
            if depth_fd in r:
                b = v4l2.v4l2_buffer()
                b.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
                b.memory = v4l2.V4L2_MEMORY_MMAP
                fcntl.ioctl(depth_fd, v4l2.VIDIOC_DQBUF, b)
                depth_frame_size = DEPTH_W * DEPTH_H * 10 // 8
                if b.bytesused >= depth_frame_size:
                    raw = bytes(depth_bufs[b.index][:depth_frame_size])
                    fcntl.ioctl(depth_fd, v4l2.VIDIOC_QBUF, b)
                    depth = decode_y10(raw, DEPTH_W, DEPTH_H).astype(np.float32)
                    depth[depth > max_raw] = 0
                    frame_count += 1
                    cloud_counter += 1
                    im.set_data(depth)

                    now = time.time()
                    if now - fps_t >= 1.5:
                        fps = frame_count / (now - fps_t)
                        frame_count = 0; fps_t = now
                        cy = slice(DEPTH_H//2-10, DEPTH_H//2+10)
                        cx = slice(DEPTH_W//2-16, DEPTH_W//2+16)
                        cnz = depth[cy, cx]; cnz = cnz[cnz > 0]
                        cdist = f"{cnz.mean()*DEPTH_SCALE*100:.0f}cm" if len(cnz) else "no-return"
                        title.set_text(f'{fps:.1f} fps{mode_str}  |  centre {cdist}  |  close to quit')

                    if args.show3d and cloud_counter >= cloud_every:
                        cloud_counter = 0
                        pts = depth_to_points(depth, max_m=args.max_depth)
                        if len(pts):
                            if scat3d is not None:
                                scat3d.remove()
                            # Use real colour if available, else plasma depth colourmap
                            if args.colour and latest_colour[0] is not None:
                                col_img = latest_colour[0]  # (DEPTH_H, COLOUR_W, 3)
                                cx_px = DEPTH_W / 2
                                cy_px = DEPTH_H / 2
                                h, w = depth.shape
                                ys, xs = np.mgrid[0:h, 0:w]
                                Z = depth * DEPTH_SCALE
                                valid = (Z > 0.05) & (Z < args.max_depth)
                                vx = xs[valid]; vy = ys[valid]
                                # Sample colour at same pixel (rough alignment)
                                # col_img is 640x400 so col_x = vx, col_y = vy
                                col_x = np.clip(vx, 0, COLOUR_W - 1)
                                col_y = np.clip(vy, 0, DEPTH_H - 1)
                                pt_colours = col_img[col_y, col_x].astype(float) / 255.0
                                if len(pt_colours) > 6000:
                                    idx = np.random.choice(len(pt_colours), 6000, replace=False)
                                    pts = pts[idx] if len(pts) > 6000 else pts
                                    # Re-derive pts for consistency
                                    X = (vx[idx] - cx_px) * Z[valid][idx] / DEPTH_FX
                                    Y = (vy[idx] - cy_px) * Z[valid][idx] / DEPTH_FY
                                    pts = np.column_stack([X, Y, Z[valid][idx]])
                                    pt_colours = pt_colours[idx]
                                colours = pt_colours
                            else:
                                z = pts[:, 2]
                                t = np.clip((z - 0.05) / (args.max_depth - 0.05), 0, 1)
                                import matplotlib.cm as cmx
                                colours = cmx.plasma(t)
                            scat3d = ax3d.scatter(pts[:,0], pts[:,2], -pts[:,1],
                                                  c=colours, s=0.5, alpha=0.8,
                                                  depthshade=False)
                            ax3d.set_xlim(-1, 1); ax3d.set_ylim(0, args.max_depth); ax3d.set_zlim(-0.5, 0.5)
                else:
                    fcntl.ioctl(depth_fd, v4l2.VIDIOC_QBUF, b)

            # ── Colour frame ─────────────────────────────────────────────────
            if col_fd and col_fd in r:
                b = v4l2.v4l2_buffer()
                b.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
                b.memory = v4l2.V4L2_MEMORY_MMAP
                fcntl.ioctl(col_fd, v4l2.VIDIOC_DQBUF, b)
                if b.bytesused >= COLOUR_FRAME_SIZE:
                    raw_col = bytes(col_bufs[b.index][:COLOUR_FRAME_SIZE])
                    fcntl.ioctl(col_fd, v4l2.VIDIOC_QBUF, b)
                    rgb = decode_yuyv(raw_col)
                    # Centre-crop vertically: 640x480 → 640x400
                    cropped = rgb[COLOUR_CROP_Y:COLOUR_CROP_Y + DEPTH_H, :, :]
                    # Brightness boost: stretch max pixel → 240 to cope with dark room
                    f = cropped.astype(np.float32)
                    peak = np.percentile(f, 99.5)
                    gain = (240.0 / peak) if peak > 1 else 1.0
                    boosted = np.clip(f * gain, 0, 255).astype(np.uint8)
                    latest_colour[0] = boosted
                    im_col.set_data(boosted)
                    fig.canvas.draw()  # colour frames are rare (~1fps), force full redraw
                else:
                    fcntl.ioctl(col_fd, v4l2.VIDIOC_QBUF, b)

            # ── IR frame ─────────────────────────────────────────────────────
            if ir_fd and ir_fd in r:
                b = v4l2.v4l2_buffer()
                b.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
                b.memory = v4l2.V4L2_MEMORY_MMAP
                fcntl.ioctl(ir_fd, v4l2.VIDIOC_DQBUF, b)
                if b.bytesused >= IR_FRAME_SIZE:
                    raw_ir = bytes(ir_bufs[b.index][:IR_FRAME_SIZE])
                    fcntl.ioctl(ir_fd, v4l2.VIDIOC_QBUF, b)
                    ir_img = decode_y10(raw_ir, IR_W, IR_H).astype(np.float32)
                    p99 = np.percentile(ir_img[ir_img > 0], 99) if np.any(ir_img > 0) else 1
                    im_ir.set_clim(vmin=0, vmax=max(p99, 1))
                    im_ir.set_data(ir_img)
                else:
                    fcntl.ioctl(ir_fd, v4l2.VIDIOC_QBUF, b)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        pass
    finally:
        close_stream(depth_fd, depth_bufs)
        if ir_fd:
            close_stream(ir_fd, ir_bufs)
        if col_fd:
            close_stream(col_fd, col_bufs)
        plt.close(fig)
        print("Stream closed.")


if __name__ == '__main__':
    main()
