"""Auto-detect Creality CR-Scan Otter video device nodes by per-node card name."""
import subprocess
import glob
import re


def find_otter_devices():
    """Return (depth_dev, ir_dev, colour_dev) by querying each /dev/videoN.

    Per-node card names (from v4l2-ctl --info):
      "CR-Scan Otter De..."  → Depth  (Y10, 640×400)
      "CR-Scan Otter IR..."  → IR     (Y10, 1280×800)
      "CR-Scan Otter RG..."  → Colour (YUYV, 640×480)

    Each stream has two nodes; we take the first (lowest numbered) for each type.
    Raises RuntimeError if any stream is not found.
    """
    depth = ir = colour = None

    nodes = sorted(glob.glob('/dev/video*'),
                   key=lambda p: int(re.search(r'\d+', p).group()))

    for node in nodes:
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', node, '--info'],
                capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.splitlines():
                if 'Card type' in line:
                    card = line.split(':', 1)[1].strip().upper()
                    if 'OTTER DE' in card and depth is None:
                        depth = node
                    elif 'OTTER IR' in card and ir is None:
                        ir = node
                    elif 'OTTER RG' in card and colour is None:
                        colour = node
                    break
        except (subprocess.TimeoutExpired, OSError):
            continue

    missing = [name for name, dev in [('Depth', depth), ('IR', ir), ('Colour', colour)] if dev is None]
    if missing:
        raise RuntimeError(
            f"CR-Scan Otter streams not found: {missing}\n"
            f"Check scanner is plugged in and you're in the 'plugdev' group."
        )
    return depth, ir, colour


if __name__ == '__main__':
    d, i, c = find_otter_devices()
    print(f"Depth:  {d}")
    print(f"IR:     {i}")
    print(f"Colour: {c}")
