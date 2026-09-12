# Copyright (c) 2026 grok-to-openai-api contributors.
"""Grok file upload (v2 presigned flow) + freeimage.host image hosting."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import mimetypes
import re
import socket
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote_to_bytes, urlparse

from curl_cffi import CurlMime
from curl_cffi.requests import AsyncSession, Response

from config import FREEIMAGE_API_KEY, FREEIMAGE_BASE, GROK_BASE, USER_AGENT

if TYPE_CHECKING:
    from statsig import StatsigGenerator

_HTTP_OK = 200
_HTTP_CREATED = 201
_SUCCESS_STATUSES = frozenset({_HTTP_OK, _HTTP_CREATED})


class UploadError(Exception):
    """Raised when a file upload or image hosting step fails."""


@dataclass(frozen=True)
class UploadPayload:
    """File bytes plus metadata for one Grok upload.

    Attributes:
        filename: Original file name reported to the upload API.
        data: Raw file bytes.
        mime: Optional MIME type override, guessed when None.

    """

    filename: str
    data: bytes
    mime: str | None = None


def _as_string_map(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _as_header_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, str)
    }


def guess_mime(filename: str, provided: str | None = None) -> str:
    """Return the provided MIME type or guess one from the filename.

    Args:
        filename: File name used for guessing.
        provided: Caller-supplied MIME type, preferred when usable.

    Returns:
        The MIME type to send.

    """
    if provided and "/" in provided:
        return provided
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def sig_headers(
    statsig: StatsigGenerator | None,
    path: str,
    method: str,
) -> dict[str, str]:
    """Build the x-statsig-id header when a seed pair is ready.

    The pair bootstraps from grok.com HTML, which Cloudflare may block;
    when it is unavailable the request must still go out (the pre-statsig
    upload flow worked without this header) instead of raising and
    dropping attachments.

    Args:
        statsig: Generator holding the seed pair, if any.
        path: Request path being signed.
        method: HTTP method being signed.

    Returns:
        The signing header, or empty when unavailable.

    """
    if statsig is not None and statsig.ready:
        try:
            return {"x-statsig-id": statsig.generate(path, method)}
        except (ValueError, TypeError, AttributeError, RuntimeError):
            # Corrupt seed/hex (e.g. bad cached state) must not fail the
            # upload; proceed without the optional header instead.
            return {}
    return {}


async def _request_upload_target(
    session: AsyncSession[Response],
    headers: dict[str, str],
    payload: UploadPayload,
) -> tuple[str, str, dict[str, str]]:
    """Request a presigned upload target for one file.

    Args:
        session: Authenticated HTTP session.
        headers: Base request headers with cookies and signing.
        payload: File name, bytes, and optional MIME override.

    Returns:
        The upload id, target URL, and required PUT headers.

    Raises:
        UploadError: If the init request fails or lacks target data.

    """
    mime = guess_mime(payload.filename, payload.mime)
    response = await session.post(
        f"{GROK_BASE}/rest/app-chat/upload-file-v2/init",
        headers={**headers, "content-type": "application/json"},
        json={
            "fileName": payload.filename,
            "fileMimeType": mime,
            "sizeBytes": len(payload.data),
            "multipartSupported": False,
        },
        timeout=30,
    )
    if response.status_code != _HTTP_OK:
        msg = f"init failed {response.status_code}: {response.text[:150]}"
        raise UploadError(msg)
    init = _as_string_map(response.json())
    upload_id = init.get("uploadId")
    single = _as_string_map(init.get("singlePut"))
    target_url = single.get("url")
    if (
        not isinstance(upload_id, str)
        or not upload_id
        or not isinstance(target_url, str)
        or not target_url
    ):
        msg = "init response missing uploadId/url"
        raise UploadError(msg)
    return upload_id, target_url, _as_header_map(single.get("requiredHeaders"))


async def _put_upload_bytes(
    session: AsyncSession[Response],
    target_url: str,
    required_headers: dict[str, str],
    data: bytes,
) -> None:
    """Send file bytes to the presigned target URL.

    Args:
        session: Authenticated HTTP session.
        target_url: Presigned PUT URL from the init step.
        required_headers: Extra headers demanded by the target.
        data: Raw file bytes.

    Raises:
        UploadError: If the PUT request is rejected.

    """
    put_headers = {"user-agent": USER_AGENT}
    put_headers.update(required_headers)
    response = await session.put(
        target_url,
        headers=put_headers,
        content=data,
        timeout=120,
    )
    if response.status_code not in _SUCCESS_STATUSES:
        msg = f"PUT failed {response.status_code}"
        raise UploadError(msg)


async def _complete_upload(
    session: AsyncSession[Response],
    headers: dict[str, str],
    statsig: StatsigGenerator | None,
    upload_id: str,
) -> dict[str, object]:
    """Complete the upload and return the initial file metadata.

    Args:
        session: Authenticated HTTP session.
        headers: Base request headers with cookies and signing.
        statsig: Optional generator for request signing headers.
        upload_id: Upload id from the init step.

    Returns:
        The initial file metadata mapping.

    Raises:
        UploadError: If the complete request fails.

    """
    comp_headers = {
        **headers,
        "content-type": "application/json",
        **sig_headers(statsig, "/rest/app-chat/upload-file-v2/complete", "POST"),
    }
    response = await session.post(
        f"{GROK_BASE}/rest/app-chat/upload-file-v2/complete",
        headers=comp_headers,
        json={"presigned": {"uploadId": upload_id}},
        timeout=30,
    )
    if response.status_code != _HTTP_OK:
        msg = f"complete failed {response.status_code}: {response.text[:150]}"
        raise UploadError(msg)
    completed = _as_string_map(response.json())
    return _as_string_map(completed.get("fileMetadata"))


async def _poll_upload_status(
    session: AsyncSession[Response],
    headers: dict[str, str],
    statsig: StatsigGenerator | None,
    upload_id: str,
) -> dict[str, object] | None:
    """Poll processing status until file metadata appears.

    Args:
        session: Authenticated HTTP session.
        headers: Base request headers with cookies and signing.
        statsig: Optional generator for request signing headers.
        upload_id: Upload id from the init step.

    Returns:
        The file metadata, or None when processing stalls.

    """
    for _ in range(20):
        status_headers = {
            **headers,
            **sig_headers(statsig, "/rest/app-chat/upload-file-v2/status", "GET"),
        }
        response = await session.get(
            f"{GROK_BASE}/rest/app-chat/upload-file-v2/status",
            params={"uploadId": upload_id},
            headers=status_headers,
            timeout=15,
        )
        if response.status_code == _HTTP_OK:
            body = _as_string_map(response.json())
            file_meta = _as_string_map(body.get("fileMetadata"))
            if file_meta.get("fileMetadataId"):
                return file_meta
            status = body.get("status")
            if isinstance(status, str) and status.upper() == "ERROR":
                break
        await asyncio.sleep(0.5)
    return None


async def upload_file(
    session: AsyncSession[Response],
    acc_cookie: str,
    statsig: StatsigGenerator | None,
    payload: UploadPayload,
) -> dict[str, object]:
    """Upload file bytes through the v2 presigned flow.

    Args:
        session: Authenticated HTTP session.
        acc_cookie: Account cookie header value.
        statsig: Optional generator for request signing headers.
        payload: File name, bytes, and optional MIME override.

    Returns:
        The completed file metadata from the upload API.

    Raises:
        UploadError: If any upload step fails or processing stalls.

    """
    headers = {
        "user-agent": USER_AGENT,
        "origin": GROK_BASE,
        "referer": f"{GROK_BASE}/",
        "cookie": acc_cookie,
        **sig_headers(statsig, "/rest/app-chat/upload-file-v2/init", "POST"),
    }
    upload_id, target_url, required_headers = await _request_upload_target(
        session,
        headers,
        payload,
    )
    await _put_upload_bytes(session, target_url, required_headers, payload.data)
    meta = await _complete_upload(session, headers, statsig, upload_id)
    found = await _poll_upload_status(session, headers, statsig, upload_id)
    if found is not None:
        return found
    if meta.get("fileMetadataId"):
        return meta
    msg = "file processing did not complete"
    raise UploadError(msg)


DATA_URL_RE = re.compile(r"^data:([^;,]+)?((?:;[^;,]*)*),", re.IGNORECASE)


def decode_data_url(value: str) -> tuple[bytes, str | None, str]:
    """Split a data URL into payload bytes, MIME type, and filename.

    Args:
        value: Data URL string to decode.

    Returns:
        The payload bytes, MIME type, and empty filename placeholder.

    Raises:
        UploadError: If the value is not a data URL.

    """
    match = DATA_URL_RE.match(value)
    if match is None:
        msg = "not a data URL"
        raise UploadError(msg)
    raw_mime = match.group(1)
    mime: str = (
        raw_mime
        if isinstance(raw_mime, str) and raw_mime
        else "application/octet-stream"
    )
    payload = value[match.end() :]
    if "base64" in (match.group(2) or "").lower():
        return base64.b64decode(payload), mime, ""
    return unquote_to_bytes(payload), mime, ""


# ------------------------------------------------------- freeimage.host (Chevereto v1)
DEF_FREEIMAGE_BASE = "https://freeimage.host"


def _freeimage_api_base() -> str:
    return (FREEIMAGE_BASE or DEF_FREEIMAGE_BASE).rstrip("/")


def normalize_freeimage_response(j: object) -> dict[str, object] | None:
    """Map a Chevereto v1 payload to direct-URL file metadata.

    The viewer page URL becomes viewer_url while url and display_url
    carry the direct embeddable file.

    Args:
        j: Decoded JSON response from the freeimage API.

    Returns:
        Normalized metadata, or None when the payload has no image.

    """
    if not isinstance(j, dict):
        return None
    image = j.get("image")
    if not isinstance(image, dict):
        return None
    direct = image.get("display_url") or image.get("url")
    if not isinstance(direct, str) or not direct:
        return None
    viewer = image.get("url")
    out: dict[str, object] = {
        key: item for key, item in image.items() if isinstance(key, str)
    }
    out["url"] = direct
    out["display_url"] = direct
    if isinstance(viewer, str) and viewer:
        out["viewer_url"] = viewer
    if "id" not in out:
        for key in ("id_encoded", "id", "name", "filename"):
            val = image.get(key)
            if isinstance(val, (str, int)) and val:
                out["id"] = val
                break
    return out


async def freeimage_upload(
    data: bytes,
    filename: str = "image.png",
    mime: str = "image/png",
) -> dict[str, object]:
    """Upload bytes to freeimage.host and return direct-URL metadata.

    Args:
        data: Raw image bytes.
        filename: File name reported to the API.
        mime: MIME type of the payload.

    Returns:
        Normalized file metadata with a direct URL.

    Raises:
        UploadError: If the API key is missing or the upload fails.

    """
    if not FREEIMAGE_API_KEY:
        msg = "FREEIMAGE_API_KEY not configured"
        raise UploadError(msg)
    if not data:
        msg = "freeimage upload: empty data"
        raise UploadError(msg)
    async with AsyncSession(impersonate="chrome") as http:
        form = CurlMime()
        form.addpart(name="source", filename=filename, content_type=mime, data=data)
        response = await http.post(
            # No trailing slash: the slashful URL 301-redirects and the
            # redirect drops the POST body (Chrome-method rewrite).
            f"{_freeimage_api_base()}/api/1/upload",
            params={"key": FREEIMAGE_API_KEY, "action": "upload", "format": "json"},
            multipart=form,
            timeout=60,
        )
        body: object = {}
        with suppress(ValueError, TypeError, AttributeError):
            body = response.json()
        normalized = normalize_freeimage_response(body)
        if response.status_code in _SUCCESS_STATUSES and normalized is not None:
            return normalized
        msg = f"freeimage {response.status_code}: {str(body)[:200]}"
        raise UploadError(msg)


def validate_public_url(url: str) -> None:
    """Reject non-public upload source URLs to block SSRF.

    Args:
        url: Candidate source URL.

    Raises:
        UploadError: If the URL is not a public HTTP(S) address.

    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        msg = f"unsupported URL scheme: {parsed.scheme}"
        raise UploadError(msg)
    hostname = parsed.hostname
    if not hostname:
        msg = "invalid URL hostname"
        raise UploadError(msg)
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as err:
        msg = f"DNS resolution failed for {hostname}: {err}"
        raise UploadError(msg) from err
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
            msg = f"SSRF protection: access to {ip_str} is forbidden"
            raise UploadError(msg)


async def _download_and_upload(
    http: AsyncSession[Response],
    url: str,
) -> dict[str, object]:
    response = await http.get(url, timeout=60)
    if response.status_code != _HTTP_OK or not response.content:
        msg = f"download {response.status_code}, {len(response.content)} bytes"
        raise UploadError(msg)
    content_type = response.headers.get("content-type", "image/png").split(";")[0]
    name = Path(url.split("?", maxsplit=1)[0]).name or "image"
    return await freeimage_upload(response.content, name, content_type)


async def freeimage_upload_from_url(url: str) -> dict[str, object]:
    """Upload an image by URL through freeimage.host with fallback.

    Tries server-side fetching first, then falls back to downloading
    the bytes locally and uploading them directly.

    Args:
        url: Public image URL to host.

    Returns:
        Normalized file metadata with a direct URL.

    Raises:
        UploadError: If the API key is missing or every attempt fails.

    """
    if not FREEIMAGE_API_KEY:
        msg = "FREEIMAGE_API_KEY not configured"
        raise UploadError(msg)
    validate_public_url(url)
    async with AsyncSession(impersonate="chrome") as http:
        response = await http.post(
            # Slashless endpoint, see freeimage_upload.
            f"{_freeimage_api_base()}/api/1/upload",
            data={
                "key": FREEIMAGE_API_KEY,
                "action": "upload",
                "source": url,
                "format": "json",
            },
            timeout=90,
        )
        body: object = {}
        with suppress(ValueError, TypeError, AttributeError):
            body = response.json()
        normalized = normalize_freeimage_response(body)
        if response.status_code in _SUCCESS_STATUSES and normalized is not None:
            return normalized
        # fallback: download then direct upload
        try:
            return await _download_and_upload(http, url)
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            TimeoutError,
            UploadError,
        ) as err:
            msg = (
                f"freeimage {response.status_code}: {str(body)[:160]} / download: {err}"
            )
            raise UploadError(msg) from err
