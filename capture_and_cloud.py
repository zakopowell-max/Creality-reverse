#!/usr/bin/env python3
"""
Capture depth + color from Creality CR-Scan Otter and generate point cloud.
Usage: python3 capture_and_cloud.py [--output out.ply]

Depth scale calibrated 2026-04-03:
  58.6 raw Y10 units = 30cm → scale = 0.005 m/unit
  Max 1023 units → ~5.1m (room scale confirmed by histogram gap)
"""
import v4l2, fcntl, mmap, select, struct, sys, argparse, time
import numpy as np

DEPTH_DEV = '/dev/video2'
COLOR_DEV = '/dev/video6'

DEPTH_W, DEPTH_H = 640, 400
DEPTH_FX = 620.0
DEPTH_FY = 620.0
DEPTH_CX = DEPTH_W / 2
DEPTH_CY = DEPTH_H / 2

DEPTH_SCALE = 0.005  # calibrated: 58.6 units = 30cm (2026-04-03)


def open_stream(device, width, height, pixfmt, nbuf=4):
    fd = open(device, 'rb+', buffering=0)
    fmt = v4l2.v4l2_format()
    fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    fmt.fmt.pix.width = width
    fmt.fmt.pix.height = height
    fmt.fmt.pix.pixelformat = pixfmt
    fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
    fcntl.ioctl(fd, v4l2.VIDIOC_S_FMT, fmt)
    actual_w = fmt.fmt.pix.width
    actual_h = fmt.fmt.pix.height
    pf = fmt.fmt.pix.pixelformat
    fourcc = ''.join(chr((pf >> i*8) & 0xff) for i in range(4))
    print(f"  S_FMT result: {actual_w}x{actual_h} fmt=0x{pf:08x}({fourcc!r}) bpl={fmt.fmt.pix.bytesperline} sizeimage={fmt.fmt.pix.sizeimage}")

    reqbufs = v4l2.v4l2_requestbuffers()
    reqbufs.count = nbuf
    reqbufs.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    reqbufs.memory = v4l2.V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, v4l2.VIDIOC_REQBUFS, reqbufs)
    print(f"  REQBUFS allocated: {reqbufs.count} buffers")

    buffers = []
    for i in range(reqbufs.count):
        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
        buf.index = i
        fcntl.ioctl(fd, v4l2.VIDIOC_QUERYBUF, buf)
        mm = mmap.mmap(fd.fileno(), buf.length, offset=buf.m.offset)
        buffers.append(mm)
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, buf)

    # Two-phase prime: device needs one full streaming cycle before bytesused is set correctly.
    # Phase 1: wake device up (bytesused=0 always on first open, use mmap check)
    print("  Priming scanner (phase 1: wake up)...")
    fcntl.ioctl(fd, v4l2.VIDIOC_STREAMON, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))
    p1_end = time.time() + 5.0
    while time.time() < p1_end:
        r, _, _ = select.select([fd], [], [], 1.0)
        if not r:
            break
        pbuf = v4l2.v4l2_buffer()
        pbuf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        pbuf.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, pbuf)
        raw_mm = bytes(buffers[pbuf.index][:320000])
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, pbuf)
        if any(raw_mm[i] for i in range(0, 320, 5)):
            break
    fcntl.ioctl(fd, v4l2.VIDIOC_STREAMOFF, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))

    # Phase 2: device is awake — now bytesused should work properly
    print("  Priming scanner (phase 2: wait for proper frames)...")
    for i in range(reqbufs.count):
        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
        buf.index = i
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, buf)
    fcntl.ioctl(fd, v4l2.VIDIOC_STREAMON, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))
    p2_end = time.time() + 5.0
    got_real = False
    while time.time() < p2_end:
        r, _, _ = select.select([fd], [], [], 1.0)
        if not r:
            break
        pbuf = v4l2.v4l2_buffer()
        pbuf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        pbuf.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, pbuf)
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, pbuf)
        if pbuf.bytesused >= 320000:
            got_real = True
            break
    print(f"  Prime done (got_real={got_real})")
    fcntl.ioctl(fd, v4l2.VIDIOC_STREAMOFF, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))

    # Zero mmaps and re-queue for the real capture
    for mm in buffers:
        mm.seek(0)
        mm.write(b'\x00' * len(mm))
    for i in range(reqbufs.count):
        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
        buf.index = i
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, buf)
    fcntl.ioctl(fd, v4l2.VIDIOC_STREAMON, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))
    return fd, buffers, (actual_w, actual_h)


def capture_frame(fd, buffers, timeout=5.0):
    for _ in range(200):
        r, _, _ = select.select([fd], [], [], timeout)
        if not r:
            return None
        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, buf)
        raw = bytes(buffers[buf.index][:buf.bytesused])
        # Also read full mmap — kernel may deliver data with bytesused=0 (driver quirk)
        raw_full = bytes(buffers[buf.index][:320000])
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, buf)
        if buf.bytesused > 100000:
            return raw
        if any(raw_full[i] for i in range(0, 320, 5)):  # sample every 5th byte in first 320
            return raw_full
        print(f"  (skip frame idx={buf.index} bytesused={buf.bytesused})")
    return None


def close_stream(fd):
    fcntl.ioctl(fd, v4l2.VIDIOC_STREAMOFF, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))
    fd.close()


def decode_y10_packed(raw_bytes, width, height):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.uint16)
    n = len(arr) // 5
    b = arr[:n*5].reshape(n, 5)
    p0 = (b[:,0] << 2) | (b[:,1] >> 6)
    p1 = ((b[:,1] & 0x3f) << 4) | (b[:,2] >> 4)
    p2 = ((b[:,2] & 0x0f) << 6) | (b[:,3] >> 2)
    p3 = ((b[:,3] & 0x03) << 8) | b[:,4]
    pixels = np.zeros(n * 4, dtype=np.uint16)
    pixels[0::4] = p0; pixels[1::4] = p1; pixels[2::4] = p2; pixels[3::4] = p3
    return pixels[:width * height].reshape(height, width)


def depth_to_pointcloud(depth_img, fx, fy, cx, cy, scale, min_depth=0.05, max_depth=6.0):
    h, w = depth_img.shape
    ys, xs = np.mgrid[0:h, 0:w]
    Z = depth_img.astype(float) * scale
    valid = (Z > min_depth) & (Z < max_depth)
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    return np.stack([X[valid], Y[valid], Z[valid]], axis=-1)


def save_ply(points, filename, colors=None):
    n = len(points)
    has_color = colors is not None and len(colors) == n
    with open(filename, 'wb') as f:
        header = f"ply\nformat ascii 1.0\nelement vertex {n}\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        if has_color:
            header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += "end_header\n"
        f.write(header.encode())
        for i in range(n):
            line = f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f}"
            if has_color:
                line += f" {colors[i,0]} {colors[i,1]} {colors[i,2]}"
            f.write((line + "\n").encode())
    print(f"Saved {n} points to {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='scan.ply')
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--scale', type=float, default=DEPTH_SCALE)
    parser.add_argument('--min-depth', type=float, default=0.05)
    parser.add_argument('--max-depth', type=float, default=6.0)
    args = parser.parse_args()

    print(f"Opening depth stream ({DEPTH_DEV})...")
    fd_d, bufs_d, (dw, dh) = open_stream(DEPTH_DEV, DEPTH_W, DEPTH_H, v4l2.V4L2_PIX_FMT_Y10)
    print(f"  {dw}x{dh}, scale={args.scale} m/unit → max range {1023*args.scale:.1f}m")

    print(f"Capturing {args.frames} depth frames (skipping initial empty frames)...")
    depth_sum = np.zeros((dh, dw), dtype=np.float32)
    valid_count = 0
    for i in range(args.frames):
        raw = capture_frame(fd_d, bufs_d)
        if raw:
            d = decode_y10_packed(raw, dw, dh).astype(np.float32)
            depth_sum += d
            valid_count += 1
            nz = d[d > 0]
            print(f"  Frame {i}: mean={nz.mean():.1f} ({nz.mean()*args.scale*100:.0f}cm), nonzero={len(nz)}")

    close_stream(fd_d)

    if valid_count == 0:
        print("ERROR: No valid depth frames!")
        return

    depth_avg = depth_sum / valid_count

    # Save depth PGM
    depth_norm = (depth_avg / depth_avg.max() * 255).astype(np.uint8)
    with open('/tmp/depth_avg.pgm', 'wb') as f:
        f.write(f"P5\n{dw} {dh}\n255\n".encode())
        f.write(depth_norm.tobytes())
    print("Saved /tmp/depth_avg.pgm")

    pts = depth_to_pointcloud(depth_avg, DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY,
                              args.scale, args.min_depth, args.max_depth)
    print(f"Point cloud: {len(pts)} points")
    print(f"  X=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}]"
          f"  Y=[{pts[:,1].min():.2f},{pts[:,1].max():.2f}]"
          f"  Z=[{pts[:,2].min():.2f},{pts[:,2].max():.2f}] m")

    save_ply(pts, args.output)

    # Quick matplotlib view
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#111')
    for ax in axes: ax.set_facecolor('#111')
    sc = axes[0].scatter(pts[:,0], -pts[:,1], s=0.3, c=pts[:,2], cmap='plasma', alpha=0.7)
    axes[0].set_title('Front view (depth = colour)', color='white')
    axes[0].set_xlabel('X (m)', color='#aaa'); axes[0].set_ylabel('Y (m)', color='#aaa')
    plt.colorbar(sc, ax=axes[0], label='Z depth (m)')
    axes[1].scatter(pts[:,0], pts[:,2], s=0.3, c=pts[:,2], cmap='plasma', alpha=0.7)
    axes[1].set_title('Top view', color='white')
    axes[1].set_xlabel('X (m)', color='#aaa'); axes[1].set_ylabel('Z depth (m)', color='#aaa')
    for ax in axes:
        ax.tick_params(colors='#888')
        for sp in ax.spines.values(): sp.set_edgecolor('#444')
    plt.suptitle(f'{len(pts):,} points | scale={args.scale} | {args.output}', color='white')
    plt.tight_layout()
    view_file = args.output.replace('.ply', '_view.png')
    plt.savefig(view_file, dpi=150, bbox_inches='tight', facecolor='#111')
    print(f"Saved view: {view_file}")


if __name__ == '__main__':
    main()
