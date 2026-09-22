import struct
import zlib

import numpy as np

COLORMAPS = {
    "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    "magma": [(0, 0, 4), (81, 18, 124), (183, 55, 121), (252, 137, 97), (252, 253, 191)],
    "blue": [(16, 30, 42), (47, 111, 151), (140, 224, 250)],
    "red": [(20, 8, 8), (140, 40, 30), (255, 210, 120)],
    "diverging": [(40, 90, 200), (30, 32, 40), (220, 70, 60)],
    "gray": [(0, 0, 0), (255, 255, 255)],
}


def _chunk(tag, data):
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def encode_png(arr):
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4):
        raise ValueError("不支持的图像形状: %r" % (a.shape,))
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    if a.shape[2] == 4:
        color_type = 6
    else:
        a = a[:, :, :3]
        color_type = 2
    a = np.ascontiguousarray(a)
    h, w = a.shape[0], a.shape[1]
    raw = np.zeros((h, w * a.shape[2] + 1), np.uint8)
    raw[:, 1:] = a.reshape(h, w * a.shape[2])
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
        + _chunk(b"IEND", b"")
    )


def normalize(a, lo=None, hi=None):
    a = np.asarray(a, dtype=np.float32)
    if lo is None:
        lo = float(np.nanmin(a))
    if hi is None:
        hi = float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros(a.shape, np.uint8), float(lo), float(hi)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8), float(lo), float(hi)


def colormap(u8, name="viridis"):
    pts = np.array(COLORMAPS.get(name) or COLORMAPS["viridis"], np.float32)
    n = len(pts) - 1
    t = np.asarray(u8, np.float32) / 255.0 * n
    i = np.clip(np.floor(t).astype(np.int32), 0, n - 1)
    f = (t - i)[..., None]
    return (pts[i] * (1.0 - f) + pts[i + 1] * f).astype(np.uint8)


def montage(maps, cols=None, pad=1, pad_val=0.0):
    maps = [np.asarray(m, np.float32) for m in maps]
    ch = len(maps)
    if ch == 0:
        return np.zeros((1, 1), np.float32)
    h = min(m.shape[0] for m in maps)
    w = min(m.shape[1] for m in maps)
    if cols is None:
        cols = int(np.ceil(np.sqrt(ch)))
    cols = max(1, min(cols, ch))
    rows = int(np.ceil(ch / cols))
    out = np.full((rows * (h + pad) - pad, cols * (w + pad) - pad), pad_val, np.float32)
    for k, m in enumerate(maps):
        r, c = divmod(k, cols)
        out[r * (h + pad):r * (h + pad) + h, c * (w + pad):c * (w + pad) + w] = m[:h, :w]
    return out


def to_png_data_uri(arr, cmap=None):
    import base64

    u8 = arr if np.asarray(arr).dtype == np.uint8 else normalize(arr)[0]
    if cmap:
        u8 = colormap(u8, cmap)
    return "data:image/png;base64," + base64.b64encode(encode_png(u8)).decode("ascii")
