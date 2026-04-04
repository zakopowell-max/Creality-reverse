#!/usr/bin/env python3
"""Scan Orbbec property IDs to find what this device supports."""
import ctypes, fcntl, struct, sys, time, os
sys.path.insert(0, os.path.dirname(__file__))
from devices import find_otter_devices

UVCIOC_CTRL_QUERY = 0xC0107521
UVC_GET_CUR = 0x81
UVC_SET_CUR = 0x01
XU_UNIT = 4
XU_SEL_64 = 2

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

def xu_set(fd, data: bytes):
    buf = ctypes.create_string_buffer(64)
    buf[:len(data)] = data
    q = XuQuery(unit=XU_UNIT, selector=XU_SEL_64, query=UVC_SET_CUR,
                size=64, data_ptr=ctypes.addressof(buf))
    fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, q)

def xu_get(fd) -> bytes:
    buf = ctypes.create_string_buffer(64)
    q = XuQuery(unit=XU_UNIT, selector=XU_SEL_64, query=UVC_GET_CUR,
                size=64, data_ptr=ctypes.addressof(buf))
    fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, q)
    return bytes(buf)

MAGIC_REQ = 0x4d47
OPCODE_GET = 0x0001
_seq = 0

def make_get(prop_id):
    global _seq
    _seq = (_seq + 1) & 0xFFFF
    return struct.pack('<HHHHI', MAGIC_REQ, 2, OPCODE_GET, _seq, prop_id)

# Known property names from OrbbecSDK Property.h
KNOWN = {
    3: 'LASER_BOOL', 4: 'LASER_PULSE_WIDTH', 5: 'LASER_CURRENT',
    6: 'FLOOD_BOOL', 7: 'FLOOD_LEVEL', 8: 'TEMP_COMP',
    9: 'WB_AUTO', 10: 'WB_TEMP', 11: 'SYNC_HOST',
    13: 'DEPTH_ALIGN', 14: 'DEPTH_MIRROR', 15: 'DEPTH_FLIP',
    17: 'IR_MIRROR', 18: 'IR_FLIP',
    20: 'COLOR_MIRROR', 21: 'COLOR_FLIP', 22: 'COLOR_AE',
    23: 'COLOR_AE_MODE', 24: 'COLOR_AE_MAX_EXPO',
    25: 'COLOR_GAIN', 26: 'COLOR_SATURATION', 27: 'COLOR_AUTO_WB',
    28: 'COLOR_WB', 29: 'COLOR_BRIGHTNESS', 30: 'COLOR_SHARPNESS',
    31: 'COLOR_CONTRAST', 32: 'COLOR_GAMMA', 33: 'COLOR_EXPO',
    36: 'DEPTH_GAIN', 37: 'IR_AUTO_EXPO', 38: 'IR_EXPO',
    79: 'LASER_MODE', 99: 'LASER_POWER_LEVEL',
    100: 'DEPTH_SOFT_FILTER', 106: 'LDP_BOOL', 107: 'LDP_STATUS',
    118: 'LASER_ENERGY_LEVEL', 119: 'LASER_HW_LEVEL',
    121: 'TIMER_RESET', 130: 'SYNC_SIGNAL_TRIGGER',
    148: 'LASER_OVERCURRENT_PROT', 149: 'LASER_PULSE_PROT',
    174: 'LASER_ALWAYS_ON', 175: 'LASER_ON_OFF_PATTERN',
    182: 'LASER_CONTROL',
}

DEPTH_DEV, _, _ = find_otter_devices()
fd = open(DEPTH_DEV, 'rb+', buffering=0)

print(f"Scanning properties on {DEPTH_DEV}...\n")
print(f"{'ID':>5}  {'Name':<30}  {'Error':>6}  {'cur':>12}  {'max':>12}")
print('-' * 75)

# Scan known IDs plus a range
ids_to_scan = sorted(set(list(KNOWN.keys()) + list(range(1, 50)) + list(range(70, 90)) + list(range(95, 130))))

for prop_id in ids_to_scan:
    req = make_get(prop_id)
    xu_set(fd, req)
    time.sleep(0.02)
    resp = xu_get(fd)

    magic = struct.unpack_from('<H', resp, 0)[0]
    if magic != 0x4252:
        continue

    size_hw, opcode, req_id, error = struct.unpack_from('<HHHH', resp, 2)
    payload = resp[10:10 + size_hw * 2]

    if error == 0x0002:  # unsupported
        continue

    name = KNOWN.get(prop_id, f'prop_{prop_id}')
    cur = struct.unpack_from('<i', payload, 0)[0] if len(payload) >= 4 else None
    mx  = struct.unpack_from('<i', payload, 4)[0] if len(payload) >= 8 else None
    err_str = f"0x{error:04x}"
    cur_str = str(cur) if cur is not None else 'N/A'
    mx_str  = str(mx)  if mx  is not None else 'N/A'
    print(f"{prop_id:>5}  {name:<30}  {err_str:>6}  {cur_str:>12}  {mx_str:>12}")

fd.close()
