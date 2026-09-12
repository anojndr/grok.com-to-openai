# Copyright (c) 2026 grok-to-openai-api contributors.
"""Regression tests for publishing Grok-generated images.

Grok asset URLs are private to the account that owns the user-id path. The API
must download them with that account's cookies and return only a public hosted
URL; returning the private URL makes the image disappear for API clients.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, Self, cast, overload, override
from unittest.mock import AsyncMock, patch

import pytest
from websockets.asyncio.client import ClientConnection
from websockets.frames import CloseCode
from websockets.protocol import State

import server
import uploads
from accounts import Account
from grok_gateway import GrokSession
from uploads import UploadError, normalize_freeimage_response

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Iterable

    from websockets.typing import Data, DataLike

_UPLOAD_TIMEOUT_SECONDS = 60


class FakePool:
    """Minimal pool stub exposing a fixed account snapshot."""

    def __init__(self, accounts: list[Account]) -> None:
        """Initialize the stub with a fixed account list.

        Args:
            accounts: Accounts returned by ``snapshot``.

        """
        self._accounts = accounts

    def snapshot(self) -> list[Account]:
        """Return a copy of the stubbed accounts.

        Returns:
            Snapshot list of accounts.

        """
        return list(self._accounts)


class _CheveretoFakeResp:
    """Stub response carrying a canned Chevereto payload."""

    status_code = 200

    @staticmethod
    def json() -> dict[str, object]:
        """Return a canned success payload.

        Returns:
            Chevereto-style success dict.

        """
        return {
            "status_code": 200,
            "image": {
                "url": "https://freeimage.host/i/viewer",
                "display_url": "https://iili.io/direct.png",
            },
            "status_txt": "OK",
        }


class _CheveretoFakeSession:
    """Stub session recording post arguments."""

    def __init__(self, seen: dict[str, object]) -> None:
        """Initialize the stub with a recording dict.

        Args:
            seen: Dict receiving the request URL, params, timeout, and form.

        """
        self._seen = seen

    async def __aenter__(self) -> Self:
        """Enter the stub session context.

        Returns:
            The stub session itself.

        """
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        """Exit the stub session context.

        Args:
            _exc: Exception info, ignored.

        Returns:
            False to avoid suppressing exceptions.

        """
        await asyncio.sleep(0)
        return False

    async def post(self, url: str, **kwargs: object) -> _CheveretoFakeResp:
        """Record post arguments and return a canned response.

        Args:
            url: Request URL.
            kwargs: Request keyword arguments.

        Returns:
            Canned fake response.

        """
        await asyncio.sleep(0)
        self._seen["url"] = url
        self._seen["params"] = kwargs.get("params")
        self._seen["timeout"] = kwargs.get("timeout")
        self._seen["multipart"] = kwargs.get("multipart")
        return _CheveretoFakeResp()


class _CheveretoFakeMime:
    """Stub multipart form collecting added parts."""

    def __init__(self) -> None:
        """Initialize the stub with an empty part list."""
        self.parts: list[dict[str, object]] = []

    def addpart(self, **kwargs: object) -> None:
        """Record one multipart field.

        Args:
            kwargs: Field attributes.

        """
        self.parts.append(kwargs)


def _check_chevereto_envelope(
    seen: dict[str, object],
    form: object,
) -> None:
    """Verify Chevereto request URL, params, timeout, and form.

    Args:
        seen: Recorded request URL, params, timeout, and form.
        form: Expected multipart form object posted to the API.

    """
    url_value = seen.get("url")
    if not isinstance(url_value, str):
        pytest.fail("upload URL not recorded")
    if not url_value.endswith("/api/1/upload"):
        pytest.fail(f"trailing-slash URL drops POST body: {url_value}")
    params_value = seen.get("params")
    if not isinstance(params_value, dict):
        pytest.fail("upload params not recorded")
    if params_value.get("key") != "test-key":
        pytest.fail("upload key mismatch")
    if params_value.get("action") != "upload":
        pytest.fail("upload action mismatch")
    if seen.get("multipart") is not form:
        pytest.fail("multipart form not posted")
    if seen.get("timeout") != _UPLOAD_TIMEOUT_SECONDS:
        pytest.fail("upload timeout mismatch")


def _check_chevereto_source_part(parts: list[dict[str, object]]) -> None:
    """Verify the recorded multipart source part fields.

    Args:
        parts: Recorded multipart parts.

    """
    source_parts = [part for part in parts if part.get("name") == "source"]
    if not source_parts:
        pytest.fail("multipart source part missing")
    source = source_parts[0]
    if source.get("filename") != "image.png":
        pytest.fail("multipart filename mismatch")
    if source.get("content_type") != "image/png":
        pytest.fail("multipart content type mismatch")
    if source.get("data") != b"img-bytes":
        pytest.fail("multipart data mismatch")


def _check_chevereto_request(
    seen: dict[str, object],
    parts: list[dict[str, object]],
    form: object,
) -> None:
    """Verify Chevereto request fields and recorded parts.

    Args:
        seen: Recorded request URL, params, timeout, and form.
        parts: Recorded multipart parts.
        form: Expected multipart form object posted to the API.

    """
    _check_chevereto_envelope(seen, form)
    _check_chevereto_source_part(parts)


def _find_item_event(
    payloads: list[dict[str, object]],
) -> dict[str, object]:
    """Find the conversation item-create event.

    Args:
        payloads: Recorded WebSocket payloads.

    Returns:
        The item-create event payload.

    """
    for payload in payloads:
        event = payload.get("event")
        if not isinstance(event, dict):
            continue
        if event.get("type") == "conversation.item.create":
            return payload
    pytest.fail("item create event missing")


def _first_input_chunk(item_event: dict[str, object]) -> dict[str, object]:
    """Extract the first input chunk from an item event.

    Args:
        item_event: Item-create event payload.

    Returns:
        First input chunk dict.

    """
    event_raw = item_event.get("event")
    if not isinstance(event_raw, dict):
        pytest.fail("item event malformed")
    event_value = cast("dict[str, object]", event_raw)
    item_raw = event_value.get("item")
    if not isinstance(item_raw, dict):
        pytest.fail("item payload malformed")
    item_value = cast("dict[str, object]", item_raw)
    xgrok_raw = item_value.get("x_grok")
    if not isinstance(xgrok_raw, dict):
        pytest.fail("x_grok payload malformed")
    xgrok = cast("dict[str, object]", xgrok_raw)
    chunks_raw = xgrok.get("input_chunks")
    if not isinstance(chunks_raw, list) or not chunks_raw:
        pytest.fail("input chunks missing")
    first_raw = chunks_raw[0]
    if not isinstance(first_raw, dict):
        pytest.fail("input chunk malformed")
    return cast("dict[str, object]", first_raw)


class _AttachmentMockWS(ClientConnection):
    """Stub connection recording sent payloads."""

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        """Initialize the stub in open state.

        Args:
            payloads: List receiving decoded sent payloads.

        """
        self._payloads = payloads
        self.protocol: object = SimpleNamespace(state=State.OPEN)

    @overload
    async def recv(self, decode: Literal[True]) -> str: ...

    @overload
    async def recv(self, decode: Literal[False]) -> bytes: ...

    @overload
    async def recv(self, decode: object = None) -> Data: ...

    @override
    async def recv(self, decode: object = None) -> Data:
        await asyncio.sleep(0)
        return json.dumps(
            {
                "event": {
                    "type": "response.done",
                    "response": {"status": "completed"},
                },
            },
        )

    @override
    async def send(
        self,
        message: DataLike | Iterable[DataLike] | AsyncIterable[DataLike],
        *,
        text: bool | None = None,
    ) -> None:
        await asyncio.sleep(0)
        if isinstance(message, str):
            payload: object = json.loads(message)
            if isinstance(payload, dict):
                self._payloads.append(cast("dict[str, object]", payload))
            else:
                self._payloads.append({"raw": payload})
        else:
            self._payloads.append({"raw": str(message)})

    @override
    async def close(
        self,
        code: CloseCode | int = CloseCode.NORMAL_CLOSURE,
        reason: str = "",
    ) -> None:
        await asyncio.sleep(0)


class ImageHostingTests(unittest.IsolatedAsyncioTestCase):
    """Verify private Grok assets are re-hosted publicly."""

    OWNER_UID = "owner-uid"
    ASSET_URL = "https://assets.grok.com/users/owner-uid/generated/image-id/image.jpg"

    @override
    def setUp(self) -> None:
        """Prepare owner and other accounts for each test."""
        self.owner = Account(
            index=1,
            cookies={"sso": "owner-sso", "x-userid": self.OWNER_UID},
            user_id=self.OWNER_UID,
        )
        self.other = Account(
            index=2,
            cookies={"sso": "other-sso", "x-userid": "other-uid"},
            user_id="other-uid",
        )

    async def test_private_asset_uses_owner_cookie_and_returns_public_url(
        self,
    ) -> None:
        """Verify private assets use the owner cookie and return a public URL."""
        download = AsyncMock(return_value=(b"jpeg-bytes", "image/jpeg", "image.jpg"))
        upload = AsyncMock(
            return_value={
                "url": "https://iili.io/image.jpg",
            },
        )
        fetch_by_url = AsyncMock(
            side_effect=AssertionError(
                "private Grok assets must not use unauthenticated URL fetching",
            ),
        )

        with (
            patch.object(server, "pool", FakePool([self.other, self.owner])),
            patch.object(server, "download_asset", download),
            patch.object(server, "freeimage_upload", upload),
            patch.object(server, "freeimage_upload_from_url", fetch_by_url),
        ):
            result = await server.host_images(
                [self.ASSET_URL],
                cookie=self.other.cookie_header(),
            )

        if result != ["https://iili.io/image.jpg"]:
            pytest.fail("hosted URL mismatch")
        download.assert_awaited_once_with(self.ASSET_URL, self.owner.cookie_header())
        upload.assert_awaited_once_with(b"jpeg-bytes", "image.jpg", "image/jpeg")
        fetch_by_url.assert_not_awaited()

    async def test_private_asset_is_not_returned_when_hosting_fails(self) -> None:
        """Verify failed hosting yields no public URL."""
        download = AsyncMock(side_effect=UploadError("download 403"))
        upload = AsyncMock()

        with (
            patch.object(server, "pool", FakePool([self.owner])),
            patch.object(server, "download_asset", download),
            patch.object(server, "freeimage_upload", upload),
        ):
            result = await server.host_images([self.ASSET_URL])
            if result != []:
                pytest.fail("expected no hosted URLs")


class FreeImageUploadTests(unittest.IsolatedAsyncioTestCase):
    """Verify freeimage.host upload normalization and field mapping."""

    @staticmethod
    async def test_normalize_prefers_direct_display_url() -> None:
        """Verify normalization prefers the direct display URL."""
        out = normalize_freeimage_response(
            {
                "status_code": 200,
                "image": {
                    "url": "https://freeimage.host/i/viewer",
                    "display_url": "https://iili.io/direct.png",
                    "id_encoded": "abc123",
                },
                "status_txt": "OK",
            },
        )
        if out is None:
            pytest.fail("expected normalized response")
        if out["url"] != "https://iili.io/direct.png":
            pytest.fail("direct URL mismatch")
        if out["viewer_url"] != "https://freeimage.host/i/viewer":
            pytest.fail("viewer URL mismatch")

    @staticmethod
    async def test_normalize_rejects_error_shape() -> None:
        """Verify error shapes normalize to None."""
        if (
            normalize_freeimage_response(
                {"status_code": 400, "error": {"message": "bad"}, "status_txt": "Bad"},
            )
            is not None
        ):
            pytest.fail("expected None for error shape")

    @staticmethod
    async def test_upload_posts_chevereto_fields_and_returns_direct_url() -> None:
        """Verify uploads post Chevereto fields and return the direct URL."""
        seen: dict[str, object] = {}
        fake_mime = _CheveretoFakeMime()
        with (
            patch.object(
                uploads,
                "AsyncSession",
                return_value=_CheveretoFakeSession(seen),
            ),
            patch.object(uploads, "CurlMime", return_value=fake_mime),
            patch.object(uploads, "FREEIMAGE_API_KEY", "test-key"),
        ):
            info = await uploads.freeimage_upload(
                b"img-bytes",
                "image.png",
                "image/png",
            )
        if info["url"] != "https://iili.io/direct.png":
            pytest.fail("upload URL mismatch")
        _check_chevereto_request(seen, fake_mime.parts, fake_mime)

    @staticmethod
    async def test_upload_without_key_raises() -> None:
        """Verify missing API keys raise upload errors."""
        with patch.object(uploads, "FREEIMAGE_API_KEY", ""):
            with pytest.raises(UploadError):
                await uploads.freeimage_upload(b"x", "image.png", "image/png")
            with pytest.raises(UploadError):
                await uploads.freeimage_upload_from_url("https://example.com/a.png")


class GatewayAttachmentMentionTests(unittest.IsolatedAsyncioTestCase):
    """Verify gateway attachment chunks use the file-mention field."""

    @staticmethod
    async def test_attachment_chunk_uses_file_mention_proto_field() -> None:
        """Verify ask transmits file mentions over WebSocket."""
        sess = GrokSession("ck", "uid")
        sess.conversation_id = "conv-test"
        sent_payloads: list[dict[str, object]] = []
        sess.ws = _AttachmentMockWS(sent_payloads)

        async for _ in sess.ask("prompt text", attachment_ids=["file-abc-123"]):
            pass

        first_chunk = _first_input_chunk(_find_item_event(sent_payloads))
        expected: dict[str, object] = {
            "mention": {"file_mention": {"file_id": "file-abc-123"}},
        }
        if first_chunk != expected:
            pytest.fail("file mention payload mismatch")
        mention = first_chunk.get("mention")
        if not isinstance(mention, dict):
            pytest.fail("mention payload malformed")
        if "target" in mention:
            pytest.fail("mention must not contain target")


if __name__ == "__main__":
    unittest.main()
