#!/usr/bin/env python3
"""
Test whether STREAMOFF+STREAMON between frames re-triggers frame delivery.
Hypothesis: scanner is a 'request-frame' device needing a streaming restart per frame.
"""
import v4l2, fcntl, mmap, select, struct, time
import numpy as np
from devices import find_otter_devices

DEPTH_DEV, _, _ = find_otter_devices()
DEPTH_W, DEPTH_H = 640, 400
NBUF = 4

def set_fmt(fd):
    fmt = v4l2.v4l2_format()
    fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    fmt.fmt.pix.width = DEPTH_W
    fmt.fmt.pix.height = DEPTH_H
    fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_Y10
    fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
    fcntl.ioctl(fd, v4l2.VIDIOC_S_FMT, fmt)

def alloc_buffers(fd):
    reqbufs = v4l2.v4l2_requestbuffers()
    reqbufs.count = NBUF
    reqbufs.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    reqbufs.memory = v4l2.V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, v4l2.VIDIOC_REQBUFS, reqbufs)
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
    return buffers

def free_buffers(fd, buffers):
    for mm in buffers:
        mm.close()
    reqbufs = v4l2.v4l2_requestbuffers()
    reqbufs.count = 0
    reqbufs.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    reqbufs.memory = v4l2.V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, v4l2.VIDIOC_REQBUFS, reqbufs)

def streamon(fd):
    fcntl.ioctl(fd, v4l2.VIDIOC_STREAMON, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))

def streamoff(fd):
    fcntl.ioctl(fd, v4l2.VIDIOC_STREAMOFF, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))

def drain_and_requeue(fd, buffers, timeout=3.0):
    """Drain all queued buffers, return mmap contents, requeue all."""
    frames = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        r, _, _ = select.select([fd], [], [], 0.5)
        if not r:
            break
        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, buf)
        raw = bytes(buffers[buf.index][:320000])
        frames.append((buf.index, buf.bytesused, raw))
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, buf)
    return frames

def frame_stats(raw):
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
    n = len(arr) // 5
    b = arr[:n*5].reshape(n, 5)
    p0 = (b[:,0] << 2) | (b[:,1] >> 6)
    p1 = ((b[:,1] & 0x3f) << 4) | (b[:,2] >> 4)
    p2 = ((b[:,2] & 0x0f) << 6) | (b[:,3] >> 2)
    p3 = ((b[:,3] & 0x03) << 8) | b[:,4]
    pixels = np.zeros(n * 4, dtype=np.uint16)
    pixels[0::4] = p0; pixels[1::4] = p1; pixels[2::4] = p2; pixels[3::4] = p3
    img = pixels[:DEPTH_W * DEPTH_H]
    nz = img[img > 0]
    return len(nz), nz.mean() if len(nz) else 0

# ---- PRIME ----
print("=== PRIME PASS ===")
fd_p = open(DEPTH_DEV, 'rb+', buffering=0)
set_fmt(fd_p)
bufs_p = alloc_buffers(fd_p)
streamon(fd_p)
for attempt in range(30):
    r, _, _ = select.select([fd_p], [], [], 1.0)
    if not r:
        print(f"  prime: timeout at attempt {attempt}")
        break
    buf = v4l2.v4l2_buffer()
    buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    buf.memory = v4l2.V4L2_MEMORY_MMAP
    fcntl.ioctl(fd_p, v4l2.VIDIOC_DQBUF, buf)
    raw = bytes(bufs_p[buf.index][:320000])
    nz, mean = frame_stats(raw)
    print(f"  prime frame {attempt}: bytesused={buf.bytesused} nonzero={nz} mean={mean:.1f}")
    fcntl.ioctl(fd_p, v4l2.VIDIOC_QBUF, buf)
    if nz > 10000:
        print("  → device awake, stopping prime")
        break
streamoff(fd_p)
free_buffers(fd_p, bufs_p)
fd_p.close()
print()

# ---- REAL OPEN ----
print("=== REAL OPEN ===")
fd = open(DEPTH_DEV, 'rb+', buffering=0)
set_fmt(fd)
buffers = alloc_buffers(fd)

# ---- TEST A: just streaming, many frames ----
print("\n--- TEST A: Continuous streaming (10 frames, 5s each timeout) ---")
streamon(fd)
for i in range(10):
    r, _, _ = select.select([fd], [], [], 5.0)
    if not r:
        print(f"  Frame {i}: TIMEOUT")
        continue
    buf = v4l2.v4l2_buffer()
    buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    buf.memory = v4l2.V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, buf)
    raw = bytes(buffers[buf.index][:320000])
    nz, mean = frame_stats(raw)
    print(f"  Frame {i}: bytesused={buf.bytesused} nonzero={nz} mean={mean:.1f}")
    fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, buf)
streamoff(fd)
print()

# ---- TEST B: STREAMOFF+STREAMON between each frame ----
print("\n--- TEST B: STREAMOFF+STREAMON between frames ---")
# Re-queue all buffers first
for i in range(NBUF):
    buf = v4l2.v4l2_buffer()
    buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    buf.memory = v4l2.V4L2_MEMORY_MMAP
    buf.index = i
    try:
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, buf)
    except:
        pass  # may already be queued

for i in range(5):
    streamon(fd)
    time.sleep(0.5)  # give device time to respond
    r, _, _ = select.select([fd], [], [], 5.0)
    if not r:
        print(f"  Frame {i}: TIMEOUT after STREAMON")
        streamoff(fd)
        continue
    buf = v4l2.v4l2_buffer()
    buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    buf.memory = v4l2.V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, buf)
    raw = bytes(buffers[buf.index][:320000])
    nz, mean = frame_stats(raw)
    print(f"  Frame {i}: bytesused={buf.bytesused} nonzero={nz} mean={mean:.1f}")
    fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, buf)
    streamoff(fd)
print()

# ---- TEST C: Full close/prime/open per frame ----
print("\n--- TEST C: Full close/prime/open per frame (slow but definitive) ---")
free_buffers(fd, buffers)
fd.close()

def get_one_frame():
    """Prime + open + grab one frame."""
    fd_p = open(DEPTH_DEV, 'rb+', buffering=0)
    set_fmt(fd_p)
    bufs_p = alloc_buffers(fd_p)
    streamon(fd_p)
    got_prime = False
    for _ in range(30):
        r, _, _ = select.select([fd_p], [], [], 1.0)
        if not r:
            break
        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(fd_p, v4l2.VIDIOC_DQBUF, buf)
        raw = bytes(bufs_p[buf.index][:320000])
        nz, _ = frame_stats(raw)
        fcntl.ioctl(fd_p, v4l2.VIDIOC_QBUF, buf)
        if nz > 10000:
            got_prime = True
            break
    streamoff(fd_p)
    free_buffers(fd_p, bufs_p)
    fd_p.close()
    if not got_prime:
        return None, 0, 0

    fd = open(DEPTH_DEV, 'rb+', buffering=0)
    set_fmt(fd)
    bufs = alloc_buffers(fd)
    streamon(fd)
    result = None
    for _ in range(10):
        r, _, _ = select.select([fd], [], [], 5.0)
        if not r:
            break
        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, buf)
        raw = bytes(bufs[buf.index][:320000])
        nz, mean = frame_stats(raw)
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, buf)
        if nz > 10000:
            result = (raw, nz, mean)
            break
    streamoff(fd)
    free_buffers(fd, bufs)
    fd.close()
    return result

for i in range(3):
    print(f"  Frame {i}...", end=' ', flush=True)
    result = get_one_frame()
    if result and result[0]:
        raw, nz, mean = result
        print(f"nonzero={nz} mean={mean:.1f}")
    else:
        print("FAILED")
