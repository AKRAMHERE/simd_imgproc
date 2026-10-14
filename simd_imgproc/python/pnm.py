"""Tiny binary PGM/PPM reader and writer (maxval 255), shared by the Python tools."""
import numpy as np


def read_pnm(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    tokens = []
    while len(tokens) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while data[pos:pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    pos += 1  # single whitespace after maxval
    magic, w, h, maxv = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    assert maxv == 255
    ch = {b"P5": 1, b"P6": 3}[magic]
    arr = np.frombuffer(data, dtype=np.uint8, count=w * h * ch, offset=pos)
    return arr.reshape(h, w) if ch == 1 else arr.reshape(h, w, 3)


def write_pnm(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    magic = b"P5" if arr.ndim == 2 else b"P6"
    with open(path, "wb") as f:
        f.write(magic + b"\n%d %d\n255\n" % (w, h))
        f.write(arr.tobytes())
