"""Pure-Python x-statsig-id generator for grok.com REST endpoints.

Algorithm (reversed & byte-exact verified against grok's JS):

    number = floor(now_unix) - 1682924400
    input  = METHOD + "!" + PATH + "!" + number + "obfiowerehiring" + HEX
    sha    = SHA-256(input)
    tail   = uint32LE(number) ++ sha[0:16] ++ [0x03]        # 21 bytes
    key    = random byte
    out[0]      = key
    out[1..48]  = seed[i] XOR key                           # 48-byte seed
    out[49..69] = tail[i] XOR key
    x-statsig-id = base64 RawStdEncoding(out)               # 70 bytes

HEX is the SVG-animation fingerprint f(seed): select curves[seed[5] % 4]
(the four BotoxFooter SVG path sets served in grok.com's RSC payload),
emulate the WebAnimation/getComputedStyle fingerprint, and hex-encode.

The active (seed, HEX) pair auto-refreshes from https://grok.com/index.
"""
from __future__ import annotations

import asyncio
import base64
import json
import hashlib
import math
import os
import re
import struct
import time

STATSIG_EPOCH = 1682924400  # 0x644f6370
SALT = "obfiowerehiring"
MARK = 0x03


# ---------------------------------------------------------------- hex helpers

def js_round(x: float) -> int:
    """JS Math.round() rounds half towards positive infinity (e.g. 2.5 -> 3, -2.5 -> -2)."""
    return math.floor(x + 0.5)


def js_num_to_hex(v: float) -> str:
    """JS Number.prototype.toString(16) matching IEEE-754 double precision."""
    if v == 0.0:
        return "-0" if math.copysign(1.0, v) < 0 else "0"
    neg = v < 0
    v = abs(v)
    ip = int(v)
    frac = v - ip
    s = format(ip, "x")
    if frac > 0:
        val_u64 = struct.unpack(">Q", struct.pack(">d", frac))[0]
        mantissa = (val_u64 & 0x000FFFFFFFFFFFFF) | 0x0010000000000000
        exponent = ((val_u64 >> 52) & 0x7FF) - 1023
        num = mantissa
        denom = 1 << (52 - exponent)
        digits = []
        rem = num
        while rem > 0:
            rem *= 16
            d = rem // denom
            digits.append(format(d, "x"))
            rem %= denom
        s += "." + "".join(digits)
    return ("-" if neg else "") + s


def js_to_fixed(v: float, prec: int = 2) -> float:
    p = 10 ** prec
    return js_round(v * p) / p


def _sample_cubic(t: float, a1: float, a2: float) -> float:
    return ((1 - 3 * a2 + 3 * a1) * t + (3 * a2 - 6 * a1)) * t * t + 3 * a1 * t


def _sample_cubic_derivative(t: float, a1: float, a2: float) -> float:
    return (3 * (1 - 3 * a2 + 3 * a1) * t + 2 * (3 * a2 - 6 * a1)) * t + 3 * a1


def cubic_bezier_y(x1: float, y1: float, x2: float, y2: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    t = x
    for _ in range(8):
        x_at_t = _sample_cubic(t, x1, x2) - x
        if abs(x_at_t) < 1e-7:
            return _sample_cubic(t, y1, y2)
        d = _sample_cubic_derivative(t, x1, x2)
        if abs(d) < 1e-7:
            break
        t -= x_at_t / d
    lo, hi = 0.0, 1.0
    t = x
    while lo < hi:
        x_at_t = _sample_cubic(t, x1, x2)
        if abs(x_at_t - x) < 1e-7:
            return _sample_cubic(t, y1, y2)
        if x > x_at_t:
            lo = t
        else:
            hi = t
        t = (hi + lo) / 2
    return _sample_cubic(t, y1, y2)


def _extract_numbers(seg: str) -> list[float]:
    return [float(m.group(0)) for m in re.finditer(r"-?\d+\.?\d*", seg)]


def compute_animation_hex(svg_path_d: str, seed: bytes) -> str:
    if len(seed) < 48:
        raise ValueError("seed must be at least 48 bytes")
    segments: list[list[float]] = []
    for part in svg_path_d[9:].split("C"):
        nums = _extract_numbers(part)
        if nums:
            segments.append(nums)
    if not segments:
        raise ValueError("no segments found in svg_path_d")
    seg_idx = seed[5] % len(segments)
    seg = segments[seg_idx]
    if len(seg) < 11:
        raise ValueError(f"segment {seg_idx} has fewer than 11 values")
    start_color = seg[0:3]
    end_color = seg[3:6]

    end_angle = math.floor(seg[6] * ((360 - 60) / 255) + 60)

    def sv(n: float, mn: float, mx: float) -> float:
        return js_to_fixed(n * ((mx - mn) / 255) + mn, 2)

    x1 = sv(seg[7], 0, 1)
    y1 = sv(seg[8], -1, 1)
    x2 = sv(seg[9], 0, 1)
    y2 = sv(seg[10], -1, 1)

    seek = js_round(((seed[24] % 16) * (seed[22] % 16) * (seed[23] % 16)) / 10) * 10
    progress = cubic_bezier_y(x1, y1, x2, y2, seek / 4096.0)

    def chan(s: float, e: float) -> int:
        v = js_round(s + (e - s) * progress)
        return max(0, min(255, int(v)))

    r = chan(start_color[0], end_color[0])
    g = chan(start_color[1], end_color[1])
    b = chan(start_color[2], end_color[2])
    angle = end_angle * progress * math.pi / 180
    c, s_ = math.cos(angle), math.sin(angle)
    values = [float(r), float(g), float(b), c, s_, -s_, c, 0.0, 0.0]
    buf = "".join(js_num_to_hex(js_to_fixed(v, 2)) for v in values)
    return re.sub(r"[.\-]", "", buf)


def curves_to_path(curve_segs: list[dict]) -> str:
    pieces = [
        f' {e["color"][0]},{e["color"][1]} {e["color"][2]},{e["color"][3]} '
        f'{e["color"][4]},{e["color"][5]} h {e["deg"]} s '
        f'{e["bezier"][0]},{e["bezier"][1]} {e["bezier"][2]},{e["bezier"][3]}'
        for e in curve_segs
    ]
    return "M 10,30 C" + " C".join(pieces)


# ------------------------------------------------------------ html extraction

def extract_meta_seed(html: str) -> str | None:
    # meta name uses a unicode dash; normalize any dash variant
    m = re.search(
        r'<meta[^>]*name=["\']grok[‐‑‒–—―-]site[‐‑‒–—―-]verification["\'][^>]*content=["\']([^"\']+)["\']',
        html, re.I,
    )
    if not m:
        m = re.search(
            r'content=["\']([^"\']+)["\'][^>]*name=["\']grok[‐‑‒–—―-]site[‐‑‒–—―-]verification["\']',
            html, re.I,
        )
    return m.group(1) if m else None


def extract_curves(html: str) -> list | None:
    marker = '\\"curves\\":['
    i = html.find(marker)
    escaped = True
    if i < 0:
        marker = '"curves":['
        i = html.find(marker)
        escaped = False
        if i < 0:
            return None
    blob = html.replace('\\"', '"') if escaped else html
    open_marker = '"curves":['
    i = blob.find(open_marker)
    if i < 0:
        return None
    # start must point at the '[' of the UNESCAPED marker; using the escaped
    # marker's length here skipped two chars and corrupted the extracted array.
    start = i + len(open_marker) - 1
    depth = 0
    j = start
    instr = False
    esc = False
    while j < len(blob):
        ch = blob[j]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        else:
            if ch == '"':
                instr = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if not depth:
                    break
        j += 1
    raw = blob[start:j + 1]
    d2 = 0
    end = None
    in2 = False
    e2 = False
    for k, ch in enumerate(raw):
        if in2:
            if e2:
                e2 = False
            elif ch == "\\":
                e2 = True
            elif ch == '"':
                in2 = False
        else:
            if ch == '"':
                in2 = True
            elif ch == "[":
                d2 += 1
            elif ch == "]":
                d2 -= 1
                if not d2:
                    end = k + 1
                    break
    trimmed = raw[:end] if end else raw
    try:
        return json.loads(trimmed)
    except Exception:
        return None



# ------------------------------------------------------------------ generator

class StatsigGenerator:
    def __init__(self):
        self._seed_b64: str | None = None
        self._seed_bytes: bytes | None = None
        self._hex: str | None = None
        self._fetched_at = 0.0
        self._lock = asyncio.Lock()

    async def ensure_pair(self, fetch_page) -> tuple[str, str]:
        """fetch_page: async callable () -> html text of grok.com/index."""
        async with self._lock:
            if self._hex and time.time() - self._fetched_at < 1800:
                return self._seed_b64, self._hex
            try:
                html = await fetch_page()
                seed_b64 = extract_meta_seed(html)
                curves = extract_curves(html)
                if seed_b64 and curves:
                    seed = base64.b64decode(seed_b64 + "==")
                    if len(curves) > 0 and len(seed) >= 48:
                        idx = seed[5] % len(curves)
                        path_d = curves_to_path(curves[idx])
                        computed_hex = compute_animation_hex(path_d, seed)
                        # Atomic state update
                        self._seed_b64 = seed_b64
                        self._seed_bytes = seed
                        self._hex = computed_hex
                        self._fetched_at = time.time()
            except Exception:
                pass
            return self._seed_b64, self._hex

    def set_pair(self, seed_b64: str, hex_str: str) -> None:
        self._seed_b64 = seed_b64
        try:
            self._seed_bytes = base64.b64decode(seed_b64 + "==")
        except Exception:
            self._seed_bytes = None
        self._hex = hex_str
        self._fetched_at = time.time()

    @property
    def ready(self) -> bool:
        return bool(self._hex and self._seed_b64)

    def generate(self, pathname: str, method: str, now_unix: int | None = None) -> str:
        if not (self._seed_b64 and self._hex):
            raise RuntimeError("statsig pair unavailable")
        seed = self._seed_bytes or base64.b64decode(self._seed_b64 + "==")
        if len(seed) < 48:
            raise RuntimeError("statsig seed must be at least 48 bytes")
        if now_unix is None:
            now_unix = int(time.time())
        number = (now_unix - STATSIG_EPOCH) & 0xFFFFFFFF
        data = f"{method}!{pathname}!{number}{SALT}{self._hex}"
        sha = hashlib.sha256(data.encode()).digest()
        key = os.urandom(1)[0]
        out = bytearray(70)
        out[0] = key
        for i in range(48):
            out[1 + i] = seed[i] ^ key
        tail = struct.pack("<I", number) + sha[:16] + bytes([MARK])
        for i in range(21):
            out[49 + i] = tail[i] ^ key
        return base64.b64encode(bytes(out)).rstrip(b"=").decode()
