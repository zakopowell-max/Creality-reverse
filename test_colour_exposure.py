#!/usr/bin/env python3
"""Test colour camera exposure — captures one frame using the proper prime+burst pattern."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import v4l2, fcntl, select
import numpy as np
from live_feed import (prime_device, open_colour_stream, close_stream,
                       decode_yuyv, COLOUR_W, COLOUR_H, COLOUR_FRAME_SIZE, COLOUR_DEV)

print(f"Colour device: {COLOUR_DEV}")
print("Priming...")
prime_device(with_colour=True)

print("Opening colour stream...")
fd, bufs = open_colour_stream()

raw = None
print("Waiting for frame...")
for _ in range(30):
    r, _, _ = select.select([fd], [], [], 2.0)
    if not r:
        print("  timeout")
        break
    b = v4l2.v4l2_buffer()
    b.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    b.memory = v4l2.V4L2_MEMORY_MMAP
    fcntl.ioctl(fd, v4l2.VIDIOC_DQBUF, b)
    print(f"  bytesused={b.bytesused} (need {COLOUR_FRAME_SIZE})")
    if b.bytesused >= COLOUR_FRAME_SIZE:
        raw = bytes(bufs[b.index][:COLOUR_FRAME_SIZE])
        fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, b)
        break
    fcntl.ioctl(fd, v4l2.VIDIOC_QBUF, b)

close_stream(fd, bufs)

if raw is None:
    print("ERROR: no colour frame captured")
    sys.exit(1)

img = decode_yuyv(raw)
y = img[:, :, 0].astype(float) * 0.299 + img[:, :, 1].astype(float) * 0.587 + img[:, :, 2].astype(float) * 0.114
print(f"Frame: {img.shape}  mean_Y={y.mean():.1f}  min={y.min():.0f}  max={y.max():.0f}")

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#111')
axes[0].imshow(img)
axes[0].set_title(f'Raw  mean_Y={y.mean():.1f}', color='white')
axes[0].axis('off')
boosted = np.clip(img.astype(float) * (128.0 / max(y.mean(), 1)), 0, 255).astype(np.uint8)
axes[1].imshow(boosted)
axes[1].set_title('Boosted to Y=128', color='white')
axes[1].axis('off')
for ax in axes: ax.set_facecolor('#111')
plt.tight_layout()
plt.savefig('/tmp/colour_test.png', dpi=150, bbox_inches='tight', facecolor='#111')
print("Saved /tmp/colour_test.png")
