#!/usr/bin/env python3
"""
Probe XU unit 4 on the depth device — read-only, no writes.

Uses UVCIOC_CTRL_QUERY ioctl to interrogate each selector:
  GET_INFO  → capability flags (which requests are supported)
  GET_LEN   → data length for that selector
  GET_CUR   → current value
"""
import ctypes, fcntl, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from devices import find_otter_devices

# ── ioctl plumbing ────────────────────────────────────────────────────────────
# _IOWR('u', 0x21, struct uvc_xu_control_query)  — 16 bytes on 64-bit
UVCIOC_CTRL_QUERY = 0xC0107521

UVC_GET_CUR  = 0x81
UVC_GET_LEN  = 0x85
UVC_GET_INFO = 0x86

class XuQuery(ctypes.Structure):
    _fields_ = [
        ('unit',     ctypes.c_uint8),
        ('selector', ctypes.c_uint8),
        ('query',    ctypes.c_uint8),
        ('_pad1',    ctypes.c_uint8),
        ('size',     ctypes.c_uint16),
        ('_pad2',    ctypes.c_uint16),
        ('data_ptr', ctypes.c_uint64),
    ]

def xu_get(fd, unit, selector, query, size):
    buf = ctypes.create_string_buffer(size)
    q = XuQuery(unit=unit, selector=selector, query=query, size=size,
                data_ptr=ctypes.addressof(buf))
    try:
        fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, q)
        return bytes(buf)
    except OSError as e:
        return None

# ── Main ──────────────────────────────────────────────────────────────────────
DEPTH_DEV, _, _ = find_otter_devices()
XU_UNIT = 4
print(f"Probing XU unit {XU_UNIT} on {DEPTH_DEV}\n")

fd = open(DEPTH_DEV, 'rb+', buffering=0)

for sel in range(1, 16):
    # GET_INFO → 1-byte capabilities
    info = xu_get(fd, XU_UNIT, sel, UVC_GET_INFO, 1)
    if info is None:
        continue
    caps = info[0]
    print(f"Selector {sel:2d}: caps=0x{caps:02x} ", end='')

    # GET_LEN → 2-byte length
    lenb = xu_get(fd, XU_UNIT, sel, UVC_GET_LEN, 2)
    if lenb is None:
        print("(no GET_LEN)")
        continue
    data_len = int.from_bytes(lenb, 'little')
    print(f"len={data_len:4d}  ", end='')

    # GET_CUR
    if caps & 0x01:  # GET supported
        cur = xu_get(fd, XU_UNIT, sel, UVC_GET_CUR, data_len)
        if cur:
            hex_str = cur[:32].hex(' ')
            print(f"cur[0:{min(32,data_len)}]= {hex_str}", end='')
    print()

fd.close()
