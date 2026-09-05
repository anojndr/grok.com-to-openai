"""Regression tests for publishing Grok-generated images.

Grok asset URLs are private to the account that owns the user-id path. The API
must download them with that account's cookies and return only a public hosted
URL; returning the private URL makes the image disappear for API clients.
"""

from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterable, Iterable
from types import SimpleNamespace
from typing import Any, Literal, Self, overload, override
from unittest.mock import AsyncMock, patch

from websockets.asyncio.client import ClientConnection
from websockets.frames import CloseCode
from websockets.protocol import State
from websockets.typing import Data, DataLike

import server
import uploads
from accounts import Account
from grok_gateway import GrokSession
from uploads import UploadError, _normalize_freeimage_response


class FakePool:
    def __init__(self, accounts: list[Account]):
        self._accounts = accounts

    def snapshot(self) -> list[Account]:
        return list(self._accounts)


class ImageHostingTests(unittest.IsolatedAsyncioTestCase):
    OWNER_UID = "owner-uid"
    ASSET_URL = "https://assets.grok.com/users/owner-uid/generated/image-id/image.jpg"

    @override
    def setUp(self):
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

    async def test_private_asset_uses_owner_cookie_and_returns_public_url(self):
        download = AsyncMock(return_value=(b"jpeg-bytes", "image/jpeg", "image.jpg"))
        upload = AsyncMock(
            return_value={
                "url": "https://iili.io/image.jpg",
            }
        )
        fetch_by_url = AsyncMock(
            side_effect=AssertionError(
                "private Grok assets must not use unauthenticated URL fetching"
            )
        )

        with (
            patch.object(server, "pool", FakePool([self.other, self.owner])),
            patch.object(server, "_download_asset", download),
            patch.object(server, "freeimage_upload", upload),
            patch.object(server, "freeimage_upload_from_url", fetch_by_url),
        ):
            result = await server.host_images(
                [self.ASSET_URL], cookie=self.other.cookie_header()
            )

        self.assertEqual(result, ["https://iili.io/image.jpg"])
        download.assert_awaited_once_with(self.ASSET_URL, self.owner.cookie_header())
        upload.assert_awaited_once_with(b"jpeg-bytes", "image.jpg", "image/jpeg")
        fetch_by_url.assert_not_awaited()

    async def test_private_asset_is_not_returned_when_hosting_fails(self):
        download = AsyncMock(side_effect=UploadError("download 403"))
        upload = AsyncMock()

        with (
            patch.object(server, "pool", FakePool([self.owner])),
            patch.object(server, "_download_asset", download),
            patch.object(server, "freeimage_upload", upload),
        ):
            result = await server.host_images([self.ASSET_URL])
            self.assertEqual(result, [])


class FreeImageUploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_normalize_prefers_direct_display_url(self):
        out = _normalize_freeimage_response(
            {
                "status_code": 200,
                "image": {
                    "url": "https://freeimage.host/i/viewer",
                    "display_url": "https://iili.io/direct.png",
                    "id_encoded": "abc123",
                },
                "status_txt": "OK",
            }
        )
        assert out is not None
        self.assertEqual(out["url"], "https://iili.io/direct.png")
        self.assertEqual(out["viewer_url"], "https://freeimage.host/i/viewer")

    async def test_normalize_rejects_error_shape(self):
        self.assertIsNone(
            _normalize_freeimage_response(
                {"status_code": 400, "error": {"message": "bad"}, "status_txt": "Bad"}
            )
        )

    async def test_upload_posts_chevereto_fields_and_returns_direct_url(self):
        seen: dict[str, Any] = {}

        class FakeResp:
            status_code = 200

            def json(self) -> dict[str, Any]:
                return {
                    "status_code": 200,
                    "image": {
                        "url": "https://freeimage.host/i/viewer",
                        "display_url": "https://iili.io/direct.png",
                    },
                    "status_txt": "OK",
                }

        class FakeSession:
            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *_: object) -> bool:
                return False

            async def post(self, url: str, **kwargs: Any) -> FakeResp:
                seen["url"] = url
                seen["params"] = kwargs.get("params")
                return FakeResp()

        class FakeMime:
            def __init__(self) -> None:
                self.parts: list[dict[str, Any]] = []

            def addpart(self, **kwargs: Any) -> None:
                self.parts.append(kwargs)

        fake_mime = FakeMime()
        with (
            patch.object(uploads, "AsyncSession", return_value=FakeSession()),
            patch("curl_cffi.CurlMime", return_value=fake_mime),
            patch.object(uploads, "FREEIMAGE_API_KEY", "test-key"),
        ):
            info = await uploads.freeimage_upload(
                b"img-bytes", "image.png", "image/png"
            )
        self.assertEqual(info["url"], "https://iili.io/direct.png")
        assert isinstance(seen.get("url"), str)
        self.assertTrue(
            seen["url"].endswith("/api/1/upload"),
            f"trailing-slash URL 301s and drops the POST body: {seen['url']}",
        )
        assert isinstance(seen.get("params"), dict)
        self.assertEqual(seen["params"].get("key"), "test-key")
        self.assertEqual(seen["params"].get("action"), "upload")
        self.assertTrue(any(p.get("name") == "source" for p in fake_mime.parts))

    async def test_upload_without_key_raises(self):
        with patch.object(uploads, "FREEIMAGE_API_KEY", ""):
            with self.assertRaises(UploadError):
                await uploads.freeimage_upload(b"x", "image.png", "image/png")
            with self.assertRaises(UploadError):
                await uploads.freeimage_upload_from_url("https://example.com/a.png")


class GatewayAttachmentMentionTests(unittest.IsolatedAsyncioTestCase):
    async def test_attachment_chunk_uses_file_mention_proto_field(self):
        """Verify ask() transmits {"mention": {"file_mention": {"file_id": ...}}} over WebSocket."""

        sess = GrokSession("ck", "uid")
        sess.conversation_id = "conv-test"

        sent_payloads: list[dict[str, Any]] = []

        class MockWS(ClientConnection):
            def __init__(self) -> None:
                self.protocol: Any = SimpleNamespace(state=State.OPEN)

            @overload
            async def recv(self, decode: Literal[True]) -> str: ...

            @overload
            async def recv(self, decode: Literal[False]) -> bytes: ...

            @overload
            async def recv(self, decode: bool | None = None) -> Data: ...

            @override
            async def recv(self, decode: bool | None = None) -> Data:
                return json.dumps(
                    {
                        "event": {
                            "type": "response.done",
                            "response": {"status": "completed"},
                        }
                    }
                )

            @override
            async def send(
                self,
                message: DataLike | Iterable[DataLike] | AsyncIterable[DataLike],
                *,
                text: bool | None = None,
            ) -> None:
                if isinstance(message, str):
                    sent_payloads.append(json.loads(message))
                else:
                    sent_payloads.append(json.loads(str(message)))

            @override
            async def close(
                self, code: CloseCode | int = CloseCode.NORMAL_CLOSURE, reason: str = ""
            ) -> None:
                return None

        sess.ws = MockWS()

        async for _ in sess.ask("prompt text", attachment_ids=["file-abc-123"]):
            pass

        item_event = next(
            (
                p
                for p in sent_payloads
                if (p.get("event") or {}).get("type") == "conversation.item.create"
            ),
            None,
        )
        self.assertIsNotNone(item_event)
        assert isinstance(item_event, dict)
        input_chunks = item_event["event"]["item"]["x_grok"]["input_chunks"]
        self.assertEqual(
            input_chunks[0], {"mention": {"file_mention": {"file_id": "file-abc-123"}}}
        )
        self.assertNotIn("target", input_chunks[0]["mention"])


if __name__ == "__main__":
    unittest.main()
