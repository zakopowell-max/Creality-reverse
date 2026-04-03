#!/usr/bin/env python3
"""Capture one IR frame from /dev/video4 and save as PNG."""
import v4l2, fcntl, mmap, select, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

IR_DEV = '/dev/video4'
IR_W, IR_H = 1280, 800
FRAME_SIZE = IR_W * IR_H * 10 // 8  # 1280000 bytes packed Y10

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

# Prime the IR device (same quirk as depth)
print("Priming IR stream...")
fd = open(IR_DEV, 'rb+', buffering=0)
fmt = v4l2.v4l2_format()
fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
fmt.fmt.pix.width = IR_W; fmt.fmt.pix.height = IR_H
fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_Y10
fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
fcntl.ioctl(fd, v4l2.VIDIOC_S_FMT, fmt)

req = v4l2.v4l2_requestbuffers()
req.count = 4; req.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE; req.memory = v4l2.V4L2_MEMORY_MMAP
fcntl.ioctl(fd, v4l2.VIDIOC_REQBUFS, req)
bufs = []
for i in range(req.count):
    b = v4l2.v4l2_buffer()
    b.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = v4l2.V4L2_MEMORY_MMAP; b.index = i
    fcntl.ioctl(fd, v4l2.VIDIOC_QUERYBUF, b)
    mm = mmap.mmap(fd.fileno(), b.length, offset=b.m.offset)
    bufs.append(mm); fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, b)

fcntl.ioctl(fd, v4l2.VIDIOC_STREAMON, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))
for _ in range(30):
    r, _, _ = select.select([fd], [], [], 1.0)
    if not r: break
    b = v4l2.v4l2_buffer()
    b.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = v4l2.V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, b); fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, b)
    if b.bytesused >= FRAME_SIZE:
        break
fcntl.ioctl(fd, v4l2.VIDIOC_STREAMOFF, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))
for mm in bufs: mm.close()
req.count = 0; fcntl.ioctl(fd, v4l2.VIDIOC_REQBUFS, req)
fd.close()

# Real capture
print("Capturing IR frames...")
fd = open(IR_DEV, 'rb+', buffering=0)
fmt.fmt.pix.width = IR_W; fmt.fmt.pix.height = IR_H
fcntl.ioctl(fd, v4l2.VIDIOC_S_FMT, fmt)
req.count = 4; fcntl.ioctl(fd, v4l2.VIDIOC_REQBUFS, req)
bufs = []
for i in range(req.count):
    b = v4l2.v4l2_buffer()
    b.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = v4l2.V4L2_MEMORY_MMAP; b.index = i
    fcntl.ioctl(fd, v4l2.VIDIOC_QUERYBUF, b)
    mm = mmap.mmap(fd.fileno(), b.length, offset=b.m.offset)
    bufs.append(mm); fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, b)
fcntl.ioctl(fd, v4l2.VIDIOC_STREAMON, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))

raw = None
for _ in range(20):
    r, _, _ = select.select([fd], [], [], 5.0)
    if not r: break
    b = v4l2.v4l2_buffer()
    b.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = v4l2.V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, b)
    print(f"  frame: bytesused={b.bytesused}")
    if b.bytesused >= FRAME_SIZE:
        raw = bytes(bufs[b.index][:FRAME_SIZE])
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, b)
        break
    fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, b)

fcntl.ioctl(fd, v4l2.VIDIOC_STREAMOFF, v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE))
for mm in bufs: mm.close()
req.count = 0; fcntl.ioctl(fd, v4l2.VIDIOC_REQBUFS, req)
fd.close()

if raw is None:
    print("ERROR: no IR frame captured")
    raise SystemExit(1)

ir = decode_y10(raw, IR_W, IR_H)
print(f"IR image: {IR_W}x{IR_H}, min={ir.min()} max={ir.max()} mean={ir[ir>0].mean():.1f}")

# Save as PNG
fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor='#111')
# Full frame
axes[0].imshow(ir, cmap='gray', vmin=ir.min(), vmax=np.percentile(ir, 99))
axes[0].set_title(f'IR frame {IR_W}x{IR_H}  min={ir.min()} max={ir.max()}', color='white')
axes[0].axis('off')
# Centre crop (same region as depth centre)
cy, cx = IR_H//2, IR_W//2
crop = ir[cy-100:cy+100, cx-160:cx+160]
axes[1].imshow(crop, cmap='gray', vmin=crop.min(), vmax=np.percentile(crop, 99))
axes[1].set_title('Centre crop', color='white')
axes[1].axis('off')
for ax in axes: ax.set_facecolor('#111')
plt.tight_layout()
plt.savefig('/tmp/ir_frame.png', dpi=150, bbox_inches='tight', facecolor='#111')
print("Saved /tmp/ir_frame.png")
