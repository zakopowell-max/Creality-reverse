#!/usr/bin/env python3
"""
Send Orbbec vendor XU commands to control the laser/projector on the depth device.

Protocol (from OrbbecSDK_v2 Protocol.hpp):
  Request:  [magic=0x4d47][sizeHWords][opcode][reqId][propertyId(4)][value(4)]
  Response: [magic=0x4252][sizeHWords][opcode][reqId][errorCode][...data...]

Usage:
  python3 xu_laser.py on      # turn laser/projector on
  python3 xu_laser.py off     # turn laser/projector off
  python3 xu_laser.py torch   # set laser mode=2 (Torch/flashlight)
  python3 xu_laser.py status  # read current laser state
"""
import ctypes, fcntl, struct, sys, time, os
sys.path.insert(0, os.path.dirname(__file__))
from devices import find_otter_devices

# ── UVCIOC_CTRL_QUERY plumbing ────────────────────────────────────────────────
UVCIOC_CTRL_QUERY = 0xC0107521
UVC_GET_CUR = 0x81
UVC_SET_CUR = 0x01

XU_UNIT      = 4
XU_SEL_64    = 2   # 64-byte selector  (small commands)
XU_SEL_512   = 1   # 512-byte selector
XU_SEL_1024  = 3   # 1024-byte selector

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

def xu_set(fd, selector, data: bytes):
    size = {XU_SEL_64: 64, XU_SEL_512: 512, XU_SEL_1024: 1024}[selector]
    buf = ctypes.create_string_buffer(size)
    buf[:len(data)] = data
    q = XuQuery(unit=XU_UNIT, selector=selector, query=UVC_SET_CUR,
                size=size, data_ptr=ctypes.addressof(buf))
    fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, q)

def xu_get(fd, selector) -> bytes:
    size = {XU_SEL_64: 64, XU_SEL_512: 512, XU_SEL_1024: 1024}[selector]
    buf = ctypes.create_string_buffer(size)
    q = XuQuery(unit=XU_UNIT, selector=selector, query=UVC_GET_CUR,
                size=size, data_ptr=ctypes.addressof(buf))
    fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, q)
    return bytes(buf)

# ── Protocol helpers ──────────────────────────────────────────────────────────
MAGIC_REQ  = 0x4d47   # "MG"
MAGIC_RESP = 0x4252   # "BR"
OPCODE_GET = 0x0001
OPCODE_SET = 0x0002

_req_id = 0

def make_set_property(prop_id: int, value: int) -> bytes:
    global _req_id
    _req_id = (_req_id + 1) & 0xFFFF
    # sizeHWords: payload after header = propertyId(4) + value(4) = 8 bytes = 4 half-words
    size_hw = 4
    return struct.pack('<HHHH II',
                      MAGIC_REQ, size_hw, OPCODE_SET, _req_id,
                      prop_id, value)

def make_get_property(prop_id: int) -> bytes:
    global _req_id
    _req_id = (_req_id + 1) & 0xFFFF
    # sizeHWords: payload = propertyId(4) = 2 half-words
    size_hw = 2
    return struct.pack('<HHHH I',
                      MAGIC_REQ, size_hw, OPCODE_GET, _req_id,
                      prop_id)

def parse_response(data: bytes):
    if len(data) < 10:
        return None, None, None
    magic, size_hw, opcode, req_id, error = struct.unpack_from('<HHHHH', data)
    if magic != MAGIC_RESP:
        return None, error, data[10:]
    payload = data[10:10 + size_hw * 2]
    return error, opcode, payload

def send_and_receive(fd, req: bytes):
    # Choose selector by request size
    if len(req) <= 64:
        tx_sel = XU_SEL_64
        rx_sel = XU_SEL_64
    elif len(req) <= 512:
        tx_sel = XU_SEL_512
        rx_sel = XU_SEL_512
    else:
        tx_sel = XU_SEL_1024
        rx_sel = XU_SEL_1024

    xu_set(fd, tx_sel, req)
    time.sleep(0.05)
    resp = xu_get(fd, rx_sel)
    return resp

# ── Property IDs (from OrbbecSDK Property.h) ─────────────────────────────────
OB_PROP_LASER_BOOL       = 3
OB_PROP_FLOOD_BOOL       = 6
OB_PROP_LASER_MODE_INT   = 79   # 1=IR Drive, 2=Torch (flashlight)
OB_PROP_LASER_POWER_INT  = 99

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    DEPTH_DEV, _, _ = find_otter_devices()
    print(f"Device: {DEPTH_DEV}")

    fd = open(DEPTH_DEV, 'rb+', buffering=0)

    if cmd == 'on':
        print("Sending LASER_BOOL = true ...")
        req = make_set_property(OB_PROP_LASER_BOOL, 1)
        resp = send_and_receive(fd, req)
        print(f"Response: {resp[:16].hex(' ')}")

    elif cmd == 'off':
        print("Sending LASER_BOOL = false ...")
        req = make_set_property(OB_PROP_LASER_BOOL, 0)
        resp = send_and_receive(fd, req)
        print(f"Response: {resp[:16].hex(' ')}")

    elif cmd == 'torch':
        print("Sending LASER_MODE = 2 (Torch) ...")
        req = make_set_property(OB_PROP_LASER_MODE_INT, 2)
        resp = send_and_receive(fd, req)
        print(f"Response: {resp[:16].hex(' ')}")
        time.sleep(0.1)
        print("Sending LASER_BOOL = true ...")
        req = make_set_property(OB_PROP_LASER_BOOL, 1)
        resp = send_and_receive(fd, req)
        print(f"Response: {resp[:16].hex(' ')}")

    elif cmd == 'status':
        for name, prop_id in [('LASER_BOOL', OB_PROP_LASER_BOOL),
                               ('FLOOD_BOOL', OB_PROP_FLOOD_BOOL),
                               ('LASER_MODE', OB_PROP_LASER_MODE_INT)]:
            req = make_get_property(prop_id)
            resp = send_and_receive(fd, req)
            error, opcode, payload = parse_response(resp)
            print(f"{name:15s}: magic=0x{resp[:2].hex()} error={resp[8:10].hex()} payload={payload[:8].hex(' ') if payload else 'N/A'}")

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python3 xu_laser.py [on|off|torch|status]")

    fd.close()

if __name__ == '__main__':
    main()
