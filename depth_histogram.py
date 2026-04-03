#!/usr/bin/env python3
"""Show histogram of depth values to help calibrate scale."""
import v4l2, fcntl, mmap, select
import numpy as np

fd = open('/dev/video2', 'rb+', buffering=0)
fmt = v4l2.v4l2_format()
fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
fmt.fmt.pix.width = 640; fmt.fmt.pix.height = 400
fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_Y10
fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
fcntl.ioctl(fd, v4l2.VIDIOC_S_FMT, fmt)

reqbufs = v4l2.v4l2_requestbuffers()
reqbufs.count = 4; reqbufs.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE; reqbufs.memory = v4l2.V4L2_MEMORY_MMAP
fcntl.ioctl(fd, v4l2.VIDIOC_REQBUFS, reqbufs)
buffers = []
for i in range(reqbufs.count):
    buf = v4l2.v4l2_buffer()
    buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE; buf.memory = v4l2.V4L2_MEMORY_MMAP; buf.index = i
    fcntl.ioctl(fd, v4l2.VIDIOC_QUERYBUF, buf)
    mm = mmap.mmap(fd.fileno(), buf.length, offset=buf.m.offset)
    buffers.append(mm)
    fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, buf)

fcntl.ioctl(fd, v4l2.VIDIOC_STREAMON, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))

# Keep reading until we get a non-empty frame
raw = b''
for _ in range(20):
    r, _, _ = select.select([fd], [], [], 5.0)
    if not r: break
    buf = v4l2.v4l2_buffer()
    buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE; buf.memory = v4l2.V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, buf)
    candidate = bytes(buffers[buf.index][:buf.bytesused])
    fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, buf)
    if len(candidate) == 320000:
        raw = candidate
        break
    print(f"  (skip empty frame, bytesused={buf.bytesused})")

if not raw:
    print("ERROR: no valid frames received")
    import sys; sys.exit(1)
arr = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
n = len(arr) // 5
b = arr[:n*5].reshape(n, 5)
pixels = np.zeros(n*4, dtype=np.uint16)
pixels[0::4] = (b[:,0]<<2)|(b[:,1]>>6)
pixels[1::4] = ((b[:,1]&0x3f)<<4)|(b[:,2]>>4)
pixels[2::4] = ((b[:,2]&0x0f)<<6)|(b[:,3]>>2)
pixels[3::4] = ((b[:,3]&0x03)<<8)|b[:,4]
img = pixels[:640*400].reshape(400, 640)

nonzero = img[img > 0]
print(f"Nonzero pixels: {len(nonzero)} / {img.size} ({100*len(nonzero)/img.size:.1f}%)")
print(f"Min={nonzero.min() if len(nonzero) else 0}, Max={img.max()}, Mean={nonzero.mean():.1f}")
print(f"\nHistogram (nonzero values):")
hist, edges = np.histogram(nonzero, bins=20)
for i, (h, lo, hi) in enumerate(zip(hist, edges[:-1], edges[1:])):
    bar = '█' * int(40 * h / hist.max())
    print(f"  {lo:5.0f}-{hi:5.0f}: {h:6d} {bar}")

print(f"\nCenter pixel region (middle 10%): {img[180:220, 288:352].mean():.1f}")
print(f"If scale=0.01: center ~= {img[180:220, 288:352].mean()*0.01:.2f}m")
print(f"If scale=1.0:  center ~= {img[180:220, 288:352].mean()*1.0:.1f} (raw units)")

fcntl.ioctl(fd, v4l2.VIDIOC_STREAMOFF, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))
fd.close()
