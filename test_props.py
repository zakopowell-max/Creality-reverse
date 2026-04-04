import sys, os, struct, time
sys.path.insert(0, os.path.dirname(__file__))
from xu_laser import make_set_property, send_and_receive
from devices import find_otter_devices

DEPTH_DEV, _, _ = find_otter_devices()
fd = open(DEPTH_DEV, 'rb+', buffering=0)

for prop_id in [81, 88, 101, 104, 108]:
    req = make_set_property(prop_id, 1)
    resp = send_and_receive(fd, req)
    err = struct.unpack_from('<H', resp, 8)[0]
    print(f'prop_{prop_id} SET=1: error=0x{err:04x}')
    time.sleep(0.5)
    req = make_set_property(prop_id, 0)
    resp = send_and_receive(fd, req)
    err = struct.unpack_from('<H', resp, 8)[0]
    print(f'prop_{prop_id} SET=0: error=0x{err:04x}')

fd.close()
