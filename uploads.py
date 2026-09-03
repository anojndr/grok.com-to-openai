"""Grok file upload (v2 presigned flow) + PixelVault image hosting."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import mimetypes
import re
import socket
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from curl_cffi.requests import AsyncSession

from config import GROK_BASE, PIXELVAULT_API_KEY, PIXELVAULT_BASE, USER_AGENT
from statsig import StatsigGenerator


class UploadError(Exception):
    pass


def guess_mime(filename: str, provided: str | None = None) -> str:
    if provided and "/" in provided:
        return provided
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _sig_headers(
    statsig: StatsigGenerator | None, path: str, method: str
) -> dict[str, str]:
    """x-statsig-id header when the generator holds a seed/hex pair; {} otherwise.

    The pair bootstraps from grok.com HTML, which Cloudflare may block; when it
    is unavailable the request must still go out (the pre-statsig upload flow
    worked without this header) instead of raising and dropping attachments.
    """
    if statsig is not None and statsig.ready:
        try:
            return {"x-statsig-id": statsig.generate(path, method)}
        except (ValueError, TypeError, AttributeError, RuntimeError):
            # Corrupt seed/hex (e.g. bad cached state) must not fail the
            # upload; proceed without the optional header instead.
            return {}
    return {}


async def upload_file(
    session: AsyncSession[Any],
    acc_cookie: str,
    statsig: StatsigGenerator | None,
    filename: str,
    data: bytes,
    mime: str | None = None,
) -> dict[str, Any]:
    """Upload via /rest/app-chat/upload-file-v2; returns file metadata dict."""
    mime = guess_mime(filename, mime)
    size = len(data)
    headers_base = {
        "user-agent": USER_AGENT,
        "origin": GROK_BASE,
        "referer": f"{GROK_BASE}/",
        "cookie": acc_cookie,
        **_sig_headers(statsig, "/rest/app-chat/upload-file-v2/init", "POST"),
    }
    r = await session.post(
        f"{GROK_BASE}/rest/app-chat/upload-file-v2/init",
        headers={**headers_base, "content-type": "application/json"},
        json={
            "fileName": filename,
            "fileMimeType": mime,
            "sizeBytes": size,
            "multipartSupported": False,
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise UploadError(f"init failed {r.status_code}: {r.text[:150]}")
    init = r.json()
    upload_id = init.get("uploadId") or ""
    single = init.get("singlePut") or {}
    url = single.get("url")
    req_headers = single.get("requiredHeaders") or {}
    if not (upload_id and url):
        raise UploadError("init response missing uploadId/url")

    put_headers = {"user-agent": USER_AGENT}
    for k, v in (req_headers or {}).items():
        put_headers[k] = v
    pr = await session.put(url, headers=put_headers, content=data, timeout=120)
    if pr.status_code not in (200, 201):
        raise UploadError(f"PUT failed {pr.status_code}")

    comp_headers = {
        **headers_base,
        "content-type": "application/json",
        **_sig_headers(statsig, "/rest/app-chat/upload-file-v2/complete", "POST"),
    }
    cr = await session.post(
        f"{GROK_BASE}/rest/app-chat/upload-file-v2/complete",
        headers=comp_headers,
        json={"presigned": {"uploadId": upload_id}},
        timeout=30,
    )
    if cr.status_code != 200:
        raise UploadError(f"complete failed {cr.status_code}: {cr.text[:150]}")
    comp = cr.json()
    meta = comp.get("fileMetadata") or {}
    # Poll processing status briefly
    fid = meta.get("fileMetadataId") or ""
    for _ in range(20):
        st_headers = {
            **headers_base,
            **_sig_headers(statsig, "/rest/app-chat/upload-file-v2/status", "GET"),
        }
        sr = await session.get(
            f"{GROK_BASE}/rest/app-chat/upload-file-v2/status",
            params={"uploadId": upload_id},
            headers=st_headers,
            timeout=15,
        )
        if sr.status_code == 200:
            sj = sr.json()
            fm = sj.get("fileMetadata")
            status = (sj.get("status") or "").upper()
            if fm and fm.get("fileMetadataId"):
                return fm
            if status == "ERROR":
                break
        await asyncio.sleep(0.5)
    if fid:
        return meta
    raise UploadError("file processing did not complete")


DATA_URL_RE = re.compile(r"^data:([^;,]+)?((?:;[^;,]*)*),", re.IGNORECASE)


def decode_data_url(value: str) -> tuple[bytes, str | None, str]:
    m = DATA_URL_RE.match(value)
    if not m:
        raise UploadError("not a data URL")
    raw_mime = m.group(1)
    mime: str = (
        raw_mime
        if isinstance(raw_mime, str) and raw_mime
        else "application/octet-stream"
    )
    payload = value[m.end() :]
    if "base64" in (m.group(2) or "").lower():
        return base64.b64decode(payload), mime, ""
    from urllib.parse import unquote_to_bytes

    return unquote_to_bytes(payload), mime, ""


# ------------------------------------------------------------------ PixelVault


async def pixelvault_upload(
    data: bytes, filename: str = "image.png", mime: str = "image/png"
) -> dict[str, Any]:
    """Upload bytes to PixelVault; returns {'id','url',...}. Raises on failure."""
    if not PIXELVAULT_API_KEY:
        raise UploadError("PIXELVAULT_API_KEY not configured")
    async with AsyncSession(impersonate="chrome") as s:
        from curl_cffi import CurlMime

        form = CurlMime()
        form.addpart(name="file", filename=filename, content_type=mime, data=data)
        r = await s.post(
            f"{PIXELVAULT_BASE}/v1/images",
            headers={"Authorization": f"Bearer {PIXELVAULT_API_KEY}"},
            multipart=form,
            timeout=60,
        )
        j = {}
        with suppress(ValueError, TypeError, AttributeError):
            j = r.json()
        if r.status_code in (200, 201) and isinstance(j.get("data"), dict):
            return j["data"]
        raise UploadError(f"pixelvault {r.status_code}: {str(j)[:200]}")


def validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UploadError(f"unsupported URL scheme: {parsed.scheme}")
    hostname = parsed.hostname
    if not hostname:
        raise UploadError("invalid URL hostname")
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for entry in addr_info:
            ip_str = entry[4][0]
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
            ):
                raise UploadError(f"SSRF protection: access to {ip_str} is forbidden")
    except socket.gaierror as e:
        raise UploadError(f"DNS resolution failed for {hostname}: {e}")


async def pixelvault_upload_from_url(url: str) -> dict[str, Any]:
    """Server-side fetch upload: POST /v1/images {url}."""
    if not PIXELVAULT_API_KEY:
        raise UploadError("PIXELVAULT_API_KEY not configured")
    validate_public_url(url)
    async with AsyncSession(impersonate="chrome") as s:
        r = await s.post(
            f"{PIXELVAULT_BASE}/v1/images",
            headers={
                "Authorization": f"Bearer {PIXELVAULT_API_KEY}",
                "content-type": "application/json",
            },
            json={"url": url},
            timeout=90,
        )
        j = {}
        with suppress(ValueError, TypeError, AttributeError):
            j = r.json()
        if r.status_code in (200, 201) and isinstance(j.get("data"), dict):
            return j["data"]
        # fallback: download then direct upload
        try:
            dr = await s.get(url, timeout=60)
            if dr.status_code != 200 or not dr.content:
                raise UploadError(f"download {dr.status_code}, {len(dr.content)} bytes")
            ct = dr.headers.get("content-type", "image/png").split(";")[0]
            name = Path(url.split("?")[0]).name or "image"
            return await pixelvault_upload(dr.content, name, ct)
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            TimeoutError,
            UploadError,
        ) as e:
            raise UploadError(
                f"pixelvault {r.status_code}: {str(j)[:160]} / download: {e}"
            ) from e
