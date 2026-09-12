# Copyright (c) 2026 grok-to-openai-api contributors.
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
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import struct
import time
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from session_store import SqliteStore

STATSIG_EPOCH = 1682924400  # 0x644f6370
SALT = "obfiowerehiring"
MARK = 0x03
_PAIR_TTL_SECONDS = 1800
_MIN_SEED_BYTES = 48
_BEZIER_TOLERANCE = 1e-7
_MIN_SEGMENT_VALUES = 11

# Dash variants matched in the verification meta name, written as escapes so
# the source stays ASCII-only (U+2010..U+2015 plus hyphen-minus).
_DASH_CLASS = r"[\u2010\u2011\u2012\u2013\u2014\u2015-]"
_META_BEFORE_RE = re.compile(
    rf"<meta[^>]*name=[\"']grok{_DASH_CLASS}site{_DASH_CLASS}"
    rf"verification[\"'][^>]*content=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
_META_AFTER_RE = re.compile(
    rf"content=[\"']([^\"']+)[\"'][^>]*name=[\"']grok{_DASH_CLASS}"
    rf"site{_DASH_CLASS}verification[\"']",
    re.IGNORECASE,
)


class _CurveSegment(TypedDict):
    color: Sequence[object]
    deg: object
    bezier: Sequence[object]


def js_round(x: float) -> int:
    """Round half towards positive infinity like JS Math.round().

    Args:
        x: Value to round.

    Returns:
        The rounded integer.

    """
    return math.floor(x + 0.5)


def js_num_to_hex(v: float) -> str:
    """Format a float like JS Number.prototype.toString(16).

    Args:
        v: Value to format with IEEE-754 double precision.

    Returns:
        The lowercase hexadecimal representation.

    """
    if not v:
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
    """Round to fixed decimals like JS Number.prototype.toFixed().

    Args:
        v: Value to round.
        prec: Decimals kept after rounding.

    Returns:
        The rounded float.

    """
    p: int = int(10**prec)
    return float(js_round(v * p)) / float(p)


def _sample_cubic(t: float, a1: float, a2: float) -> float:
    return ((1 - 3 * a2 + 3 * a1) * t + (3 * a2 - 6 * a1)) * t * t + 3 * a1 * t


def _sample_cubic_derivative(t: float, a1: float, a2: float) -> float:
    return (3 * (1 - 3 * a2 + 3 * a1) * t + 2 * (3 * a2 - 6 * a1)) * t + 3 * a1


def cubic_bezier_y(x1: float, y1: float, x2: float, y2: float, x: float) -> float:
    """Evaluate a cubic bezier curve at horizontal position x.

    Args:
        x1: First control point x.
        y1: First control point y.
        x2: Second control point x.
        y2: Second control point y.
        x: Horizontal position in the unit interval.

    Returns:
        The curve y value at x.

    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    t = x
    for _ in range(8):
        x_at_t = _sample_cubic(t, x1, x2) - x
        if abs(x_at_t) < _BEZIER_TOLERANCE:
            return _sample_cubic(t, y1, y2)
        d = _sample_cubic_derivative(t, x1, x2)
        if abs(d) < _BEZIER_TOLERANCE:
            break
        t -= x_at_t / d
    lo, hi = 0.0, 1.0
    t = x
    while lo < hi:
        x_at_t = _sample_cubic(t, x1, x2)
        if abs(x_at_t - x) < _BEZIER_TOLERANCE:
            return _sample_cubic(t, y1, y2)
        if x > x_at_t:
            lo = t
        else:
            hi = t
        t = (hi + lo) / 2
    return _sample_cubic(t, y1, y2)


def _extract_numbers(seg: str) -> list[float]:
    return [float(m.group(0)) for m in re.finditer(r"-?\d+\.?\d*", seg)]


def _select_curve_segment(svg_path_d: str, seed: bytes) -> list[float]:
    segments: list[list[float]] = []
    for part in svg_path_d[9:].split("C"):
        nums = _extract_numbers(part)
        if nums:
            segments.append(nums)
    if not segments:
        msg = "no segments found in svg_path_d"
        raise ValueError(msg)
    seg_idx = seed[5] % len(segments)
    seg = segments[seg_idx]
    if len(seg) < _MIN_SEGMENT_VALUES:
        msg = f"segment {seg_idx} has fewer than 11 values"
        raise ValueError(msg)
    return seg


def _bezier_control_points(seg: list[float]) -> tuple[float, float, float, float]:
    def scale(n: float, low: float, high: float) -> float:
        return js_to_fixed(n * ((high - low) / 255) + low, 2)

    return (
        scale(seg[7], 0, 1),
        scale(seg[8], -1, 1),
        scale(seg[9], 0, 1),
        scale(seg[10], -1, 1),
    )


def _fingerprint_values(seg: list[float], progress: float) -> list[float]:
    def channel(start: float, end: float) -> int:
        value = js_round(start + (end - start) * progress)
        return max(0, min(255, int(value)))

    red = channel(seg[0], seg[3])
    green = channel(seg[1], seg[4])
    blue = channel(seg[2], seg[5])
    end_angle = math.floor(seg[6] * ((360 - 60) / 255) + 60)
    angle = end_angle * progress * math.pi / 180
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    values = [float(red), float(green), float(blue)]
    values.extend([cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0])
    return values


def compute_animation_hex(svg_path_d: str, seed: bytes) -> str:
    """Compute the SVG-animation fingerprint hex for one seed.

    Args:
        svg_path_d: Combined SVG path data for the selected curves.
        seed: Random seed bytes of at least 48 entries.

    Returns:
        The hex fingerprint with dots and dashes stripped.

    Raises:
        ValueError: If the seed is short or the path has no segments.

    """
    if len(seed) < _MIN_SEED_BYTES:
        msg = "seed must be at least 48 bytes"
        raise ValueError(msg)
    seg = _select_curve_segment(svg_path_d, seed)
    x1, y1, x2, y2 = _bezier_control_points(seg)
    seek = js_round(((seed[24] % 16) * (seed[22] % 16) * (seed[23] % 16)) / 10) * 10
    progress = cubic_bezier_y(x1, y1, x2, y2, seek / 4096.0)
    values = _fingerprint_values(seg, progress)
    buf = "".join(js_num_to_hex(js_to_fixed(v, 2)) for v in values)
    return re.sub(r"[.\-]", "", buf)


def curves_to_path(curve_segs: list[_CurveSegment]) -> str:
    """Render curve segments as one SVG path definition.

    Args:
        curve_segs: Curve segments with color stops and handles.

    Returns:
        The combined SVG path data string.

    """
    pieces = [
        f" {entry['color'][0]},{entry['color'][1]}"
        f" {entry['color'][2]},{entry['color'][3]}"
        f" {entry['color'][4]},{entry['color'][5]}"
        f" h {entry['deg']} s"
        f" {entry['bezier'][0]},{entry['bezier'][1]}"
        f" {entry['bezier'][2]},{entry['bezier'][3]}"
        for entry in curve_segs
    ]
    return "M 10,30 C" + " C".join(pieces)


def extract_meta_seed(html: str) -> str | None:
    """Extract the base64 statsig seed from the page meta tag.

    Args:
        html: Page HTML that may embed the verification meta tag.

    Returns:
        The seed string, or None when the tag is absent.

    """
    match = _META_BEFORE_RE.search(html)
    if match is None:
        match = _META_AFTER_RE.search(html)
    if match is None:
        return None
    val = match.group(1)
    return val if isinstance(val, str) else None


def _normalized_curves_blob(html: str) -> str | None:
    if html.find('\\"curves\\":[') >= 0:
        return html.replace('\\"', '"')
    if html.find('"curves":[') >= 0:
        return html
    return None


def _scan_array_end(blob: str, start: int) -> int:
    depth = 0
    instr = False
    esc = False
    pos = start
    while pos < len(blob):
        ch = blob[pos]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        elif ch == '"':
            instr = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if not depth:
                break
        pos += 1
    return pos


def _trim_balanced_array(raw: str) -> str:
    depth = 0
    end = None
    in_string = False
    escaped = False
    for pos, ch in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if not depth:
                end = pos + 1
                break
    return raw[:end] if end else raw


def extract_curves(html: str) -> list[object] | None:
    """Extract the decoded curves array from page HTML.

    Args:
        html: Page HTML that may embed the curves payload.

    Returns:
        The decoded curves array, or None when absent or malformed.

    """
    blob = _normalized_curves_blob(html)
    if blob is None:
        return None
    # Start must point at the '[' of the UNESCAPED marker; using the escaped
    # marker's length here skipped two chars and corrupted the extracted array.
    open_marker = '"curves":['
    at = blob.find(open_marker)
    if at < 0:
        return None
    start = at + len(open_marker) - 1
    end = _scan_array_end(blob, start)
    trimmed = _trim_balanced_array(blob[start : end + 1])
    try:
        decoded = json.loads(trimmed)
    except (ValueError, TypeError, AttributeError):
        return None
    if not isinstance(decoded, list):
        return None
    items: list[object] = []
    items.extend(decoded)
    return items


def _as_curve_list(value: object) -> list[_CurveSegment] | None:
    if not isinstance(value, list):
        return None
    segments: list[_CurveSegment] = []
    for entry in value:
        if not isinstance(entry, dict):
            return None
        color = entry.get("color")
        bezier = entry.get("bezier")
        if not isinstance(color, list) or not isinstance(bezier, list):
            return None
        if "deg" not in entry:
            return None
        segments.append(
            {"color": color, "deg": entry.get("deg"), "bezier": bezier},
        )
    return segments


def _decode_seed_page(html: str | None) -> tuple[bytes, str, list[object]] | None:
    if not html:
        return None
    seed_b64 = extract_meta_seed(html)
    curves = extract_curves(html)
    if not seed_b64 or not curves:
        return None
    try:
        seed = base64.b64decode(seed_b64 + "==")
    except (ValueError, TypeError, AttributeError):
        return None
    return seed, seed_b64, curves


def _hex_for_seed(
    seed: bytes,
    seed_b64: str,
    curves: list[object],
) -> tuple[str, str] | None:
    if len(curves) == 0 or len(seed) < _MIN_SEED_BYTES:
        return None
    idx = seed[5] % len(curves)
    segments = _as_curve_list(curves[idx])
    if segments is None:
        return None
    try:
        path_d = curves_to_path(segments)
        computed_hex = compute_animation_hex(path_d, seed)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError):
        return None
    return seed_b64, computed_hex


class StatsigGenerator:
    """Generate x-statsig-id header values with an auto-refreshing pair.

    Attributes:
        store: Optional persistent cache for the seed pair.

    """

    def __init__(self, store: SqliteStore | None = None) -> None:
        """Load the cached pair when a store is provided.

        Args:
            store: Optional persistent cache for the seed pair.

        """
        self._seed_b64: str | None = None
        self._seed_bytes: bytes | None = None
        self._hex: str | None = None
        self._fetched_at = 0.0
        self._lock = asyncio.Lock()
        self._fetch_task: asyncio.Task[tuple[str, str] | None] | None = None
        self.store = store
        if self.store:
            cached = self.store.get_statsig()
            if cached:
                s_b64, h_str, f_at = cached
                self._seed_b64 = s_b64
                try:
                    self._seed_bytes = base64.b64decode(s_b64 + "==")
                except (ValueError, TypeError, AttributeError):
                    self._seed_bytes = None
                self._hex = h_str
                self._fetched_at = f_at

    @property
    def seed_b64(self) -> str | None:
        """Cached base64 seed, or None before the first fetch.

        Returns:
            The seed string, or None when no pair is cached.

        """
        return self._seed_b64

    @property
    def hex_digest(self) -> str | None:
        """Cached animation fingerprint hex, or None before fetching.

        Returns:
            The hex string, or None when no pair is cached.

        """
        return self._hex

    def _fresh_pair(self) -> tuple[str, str] | None:
        if self._hex and time.time() - self._fetched_at < _PAIR_TTL_SECONDS:
            seed_b64 = self._seed_b64
            hex_ = self._hex
            if seed_b64 and hex_:
                return seed_b64, hex_
        return None

    async def _do_fetch(
        self,
        fetch_page: Callable[[], Awaitable[str | None]],
    ) -> tuple[str, str] | None:
        try:
            html = await fetch_page()
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            TimeoutError,
        ) as err:
            logging.getLogger("uvicorn.error").debug(
                "statsig page fetch failed: %s",
                err,
            )
            html = None
        decoded = _decode_seed_page(html)
        if decoded is None:
            return None
        seed, seed_b64, curves = decoded
        pair = _hex_for_seed(seed, seed_b64, curves)
        if pair is None:
            return None
        return await self._publish_pair(seed, seed_b64, pair[1])

    async def _publish_pair(
        self,
        seed: bytes,
        seed_b64: str,
        computed_hex: str,
    ) -> tuple[str, str]:
        async with self._lock:
            if self._hex and time.time() - self._fetched_at < _PAIR_TTL_SECONDS:
                cached_seed = self._seed_b64
                cached_hex = self._hex
                if cached_seed and cached_hex:
                    return cached_seed, cached_hex
            self._seed_b64 = seed_b64
            self._seed_bytes = seed
            self._hex = computed_hex
            self._fetched_at = time.time()
            if self.store:
                try:
                    self.store.set_statsig(seed_b64, computed_hex, self._fetched_at)
                except (
                    OSError,
                    RuntimeError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    sqlite3.Error,
                ) as err:
                    logging.getLogger("uvicorn.error").debug(
                        "statsig cache write failed: %s",
                        err,
                    )
            return seed_b64, computed_hex

    async def ensure_pair(
        self,
        fetch_page: Callable[[], Awaitable[str | None]],
    ) -> tuple[str, str]:
        """Ensure a fresh seed pair, fetching the page when stale.

        Concurrent callers share one fetch task per expiry window; a
        failed fetch falls back to the stale pair or empty strings.

        Args:
            fetch_page: Async callable returning grok.com index HTML.

        Returns:
            The active seed and hex pair, or empty strings when none.

        """
        fresh = self._fresh_pair()
        if fresh is not None:
            return fresh
        # Coalesce concurrent fetches: only one fetch per expiry window
        async with self._lock:
            fresh = self._fresh_pair()
            if fresh is not None:
                return fresh
            if self._fetch_task is not None and not self._fetch_task.done():
                task = self._fetch_task
            else:
                task = asyncio.create_task(self._do_fetch(fetch_page))
                self._fetch_task = task
        try:
            result = await task
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
        ) as err:
            logging.getLogger("uvicorn.error").debug(
                "statsig fetch task failed: %s",
                err,
            )
            result = None
        # If fetch failed, return stale; if succeeded, task already updated state
        if result is None:
            async with self._lock:
                stale_seed = self._seed_b64
                stale_hex = self._hex
                if stale_seed and stale_hex:
                    return stale_seed, stale_hex
                return "", ""
        return result

    def set_pair(self, seed_b64: str, hex_str: str) -> None:
        """Override the cached pair and persist it to the store.

        Args:
            seed_b64: Base64 seed string.
            hex_str: Animation fingerprint hex string.

        """
        self._seed_b64 = seed_b64
        try:
            self._seed_bytes = base64.b64decode(seed_b64 + "==")
        except (ValueError, TypeError, AttributeError):
            self._seed_bytes = None
        self._hex = hex_str
        self._fetched_at = time.time()
        if self.store:
            self.store.set_statsig(seed_b64, hex_str, self._fetched_at)

    @property
    def ready(self) -> bool:
        """Pair readiness for request signing.

        Returns:
            True when both seed and hex are cached.

        """
        return bool(self._hex and self._seed_b64)

    def generate(
        self,
        pathname: str,
        method: str,
        now_unix: int | None = None,
    ) -> str:
        """Generate the x-statsig-id header value for one request.

        Args:
            pathname: Request path being signed.
            method: HTTP method being signed.
            now_unix: Timestamp override, or now when omitted.

        Returns:
            The base64 header value for this request.

        Raises:
            RuntimeError: If no pair is cached or the seed is short.

        """
        if not (self._seed_b64 and self._hex):
            msg = "statsig pair unavailable"
            raise RuntimeError(msg)
        seed = self._seed_bytes or base64.b64decode(self._seed_b64 + "==")
        if len(seed) < _MIN_SEED_BYTES:
            msg = "statsig seed must be at least 48 bytes"
            raise RuntimeError(msg)
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
