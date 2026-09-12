# Copyright (c) 2026 grok-to-openai-api contributors.
"""Regression tests: attachment uploads must never be silently dropped.

2026-08-25 incident ("fact check this" + image): the statsig seed/hex pair
could not bootstrap (grok.com HTML served a Cloudflare challenge), so
statsig.generate raised before any HTTP call, stream_session_turn swallowed
the error into attachment_ids=None, and grok confidently answered as if no
image had been attached.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast, override
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.responses import StreamingResponse

import server
from accounts import Account, AccountPool
from grok_gateway import GatewayError, GrokSession, TurnResult
from session_store import SqliteStore
from statsig import StatsigGenerator
from uploads import UploadError, sig_headers

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import Request

SEED_B64 = base64.b64encode(bytes(range(48))).decode()


class SigHeaderTests(unittest.TestCase):
    """Cover statsig header omission and inclusion."""

    @staticmethod
    def test_omitted_when_generator_not_ready() -> None:
        """Verify headers are omitted when the generator is not ready."""
        gen = StatsigGenerator()
        if gen.ready:
            pytest.fail("generator should not be ready")
        if sig_headers(gen, "/rest/app-chat/x", "POST") != {}:
            pytest.fail("headers should be empty")
        if sig_headers(None, "/rest/app-chat/x", "POST") != {}:
            pytest.fail("None generator should give empty headers")

    @staticmethod
    def test_included_when_ready() -> None:
        """Verify headers are included when the generator is ready."""
        gen = StatsigGenerator()
        gen.set_pair(SEED_B64, "deadbeef")
        header = sig_headers(gen, "/rest/app-chat/x", "POST")
        if "x-statsig-id" not in header:
            pytest.fail("missing statsig header")
        if not header["x-statsig-id"]:
            pytest.fail("empty statsig header")


class FakeSess(GrokSession):
    """Record the last ask kwargs for upload assertions."""

    def __init__(self) -> None:
        """Initialise fake session."""
        super().__init__("", "", "fast")
        self.cookie_header = "ck"
        self.last_kwargs: dict[str, list[str] | str | None] | None = None

    @override
    async def ask(
        self,
        prompt: str,
        *,
        attachment_ids: list[str] | None = None,
        system_prompt: str | None = None,
        user_text: str = "",
        idle_timeout: float = 120.0,
        max_turn_timeout: float = 300.0,
    ) -> AsyncIterator[dict[str, object]]:
        self.last_kwargs = {
            "attachment_ids": attachment_ids,
            "system_prompt": system_prompt,
        }
        yield {"type": "done", "result": SimpleNamespace(text="ok")}


class StreamTurnUploadTests(unittest.IsolatedAsyncioTestCase):
    """Cover upload failure handling in stream_session_turn."""

    @staticmethod
    async def test_raises_gateway_error_when_every_upload_fails() -> None:
        """Verify a fully failed upload batch raises a gateway error."""
        sess = FakeSess()
        jobs: list[dict[str, bytes | str]] = [
            {"name": "image.png", "data": b"xx", "mime": "image/png"},
        ]
        with (
            patch.object(
                server,
                "upload_file",
                new=AsyncMock(side_effect=UploadError("init failed 403")),
            ),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
            pytest.raises(GatewayError) as exc_info,
        ):
            async for _ in server.stream_session_turn(
                sess,
                "fact check this",
                file_jobs=jobs,
            ):
                pass
        if "attachment upload failed" not in str(exc_info.value):
            pytest.fail("wrong failure message")
        if sess.last_kwargs is not None:
            pytest.fail("failed turn should not reach gateway")

    @staticmethod
    async def test_partial_failure_still_sends_uploaded_files() -> None:
        """Verify partial failure still sends uploaded files."""
        sess = FakeSess()
        jobs: list[dict[str, bytes | str]] = [
            {"name": "a.png", "data": b"a", "mime": "image/png"},
            {"name": "b.png", "data": b"b", "mime": "image/png"},
        ]
        up = AsyncMock(
            side_effect=[{"fileMetadataId": "fid-ok"}, UploadError("put failed 500")],
        )
        with (
            patch.object(server, "upload_file", new=up),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            events = [
                ev
                async for ev in server.stream_session_turn(
                    sess,
                    "prompt",
                    file_jobs=jobs,
                )
            ]
        if not any(e["type"] == "done" for e in events):
            pytest.fail("partial success missing done")
        if not isinstance(sess.last_kwargs, dict):
            pytest.fail("missing last kwargs")
        if sess.last_kwargs["attachment_ids"] != ["fid-ok"]:
            pytest.fail("partial upload ids wrong")

    @staticmethod
    async def test_no_files_passes_through_untouched() -> None:
        """Verify a file-less turn passes through untouched."""
        sess = FakeSess()
        with (
            patch.object(server, "upload_file", new=AsyncMock()) as up,
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            events = [ev async for ev in server.stream_session_turn(sess, "hello")]
        up.assert_not_awaited()
        if not any(e["type"] == "done" for e in events):
            pytest.fail("hello turn missing done")

    @staticmethod
    async def test_preexisting_ids_do_not_rescue_failed_uploads() -> None:
        """A pass-through file_id must not mask a fully-failed upload batch."""
        sess = FakeSess()
        jobs: list[dict[str, bytes | str]] = [
            {"file_id": "pre-existing"},
            {"name": "image.png", "data": b"x", "mime": "image/png"},
        ]
        with (
            patch.object(
                server,
                "upload_file",
                new=AsyncMock(side_effect=UploadError("init failed 403")),
            ),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
            pytest.raises(GatewayError),
        ):
            async for _ in server.stream_session_turn(
                sess,
                "fact check this",
                attachment_ids=["caller-supplied"],
                file_jobs=jobs,
            ):
                pass
        if sess.last_kwargs is not None:
            pytest.fail("failed turn should not reach gateway")

    @staticmethod
    async def test_upload_without_metadata_id_counts_as_failure() -> None:
        """A silent no-fileMetadataId response is a dropped file, not success."""
        sess = FakeSess()
        jobs: list[dict[str, bytes | str]] = [
            {"name": "image.png", "data": b"x", "mime": "image/png"},
        ]
        with (
            patch.object(server, "upload_file", new=AsyncMock(return_value={})),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
            pytest.raises(GatewayError) as exc_info,
        ):
            async for _ in server.stream_session_turn(
                sess,
                "prompt",
                file_jobs=jobs,
            ):
                pass
        if "no fileMetadataId" not in str(exc_info.value):
            pytest.fail("wrong failure message")

    @staticmethod
    def test_sig_headers_survive_generate_crash() -> None:
        """Verify sig headers survive a generate crash."""
        gen = StatsigGenerator()
        gen.set_pair(base64.b64encode(b"short").decode(), "hex")  # <48-byte seed
        if not gen.ready:
            pytest.fail("generator should be ready")
        with patch.object(
            gen,
            "generate",
            side_effect=RuntimeError("statsig seed must be at least 48 bytes"),
        ):
            if sig_headers(gen, "/p", "POST") != {}:
                pytest.fail("crashed generate should give empty headers")


class EnsurePairResilienceTests(unittest.IsolatedAsyncioTestCase):
    """Cover statsig pair resilience on blocked pages."""

    @staticmethod
    async def test_blocked_page_keeps_generator_unready_without_raising() -> None:
        """Verify a blocked page keeps the generator unready."""
        gen = StatsigGenerator()
        calls: list[int] = []

        async def fetch_page() -> str | None:
            await asyncio.sleep(0)
            calls.append(1)
            return None

        await gen.ensure_pair(fetch_page)
        if gen.ready:
            pytest.fail("blocked page should not ready generator")
        if len(calls) != 1:
            pytest.fail("fetch page not called once")


class ResponsesStreamFailureTests(unittest.IsolatedAsyncioTestCase):
    """Stream response failures correctly.

    A turn failure after initial SSE frames must yield response.failed,
    not abort the connection (review finding on responses_api sse()).
    """

    @override
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.store = SqliteStore(Path(self.tmp_dir) / "t.db")
        self._orig_store = server.store
        server.store = self.store
        self.fake_acc = Account(
            index=1,
            cookies={"sso": "tok", "x-userid": "uid-1"},
            user_id="uid-1",
        )
        self._orig_pool = server.pool
        pool = AccountPool(Path(self.tmp_dir) / "accounts.txt")
        pool.replace_accounts([self.fake_acc])
        server.pool = pool
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.addCleanup(setattr, server, "store", self._orig_store)
        self.addCleanup(setattr, server, "pool", self._orig_pool)
        self.addCleanup(server.SESSIONS.clear)

    def _seed_prev(self) -> server.SessionState:
        fake_grok = FakeSess()
        fake_grok.conversation_id = "conv-1"
        fake_grok.last_parent_response_id = "parent-1"
        st = server.SessionState(
            account_key=self.fake_acc.key,
            grok=fake_grok,
            user_chain=["first question"],
        )
        server.SESSIONS["resp_prev"] = st
        return st

    @staticmethod
    async def _collect(req: object) -> tuple[object, str]:
        resp = await server.responses_api(cast("Request", req))
        if not isinstance(resp, StreamingResponse):
            pytest.fail("expected streaming response")
        chunks: list[str] = [
            c if isinstance(c, str) else bytes(c).decode("utf-8")
            async for c in resp.body_iterator
        ]
        return resp, "".join(chunks)

    async def test_gateway_error_streams_response_failed_event(self) -> None:
        """Verify a gateway error streams a response failed event."""
        self._seed_prev()

        async def failing_stream(
            *_args: object,
            **_kwargs: object,
        ) -> AsyncIterator[dict[str, object]]:
            await asyncio.sleep(0)
            kind = "upstream"
            msg = "attachment upload failed: image: init failed 403"
            raise GatewayError(kind, msg)
            yield {"type": "done", "result": SimpleNamespace(text="ok")}

        class FakeRequest:
            def __init__(self, body: dict[str, object]) -> None:
                self._body: dict[str, object] = body
                self.headers: dict[str, str] = {}

            async def json(self) -> dict[str, object]:
                return self._body

        req = FakeRequest(
            {"input": "follow up", "previous_response_id": "resp_prev", "stream": True},
        )
        with (
            patch.object(server, "stream_session_turn", new=failing_stream),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
            patch.object(server, "host_images", new=AsyncMock(return_value=[])),
        ):
            _resp, text = await self._collect(req)

        if "response.failed" not in text:
            pytest.fail("missing response.failed")
        if "attachment upload failed" not in text:
            pytest.fail("missing upload failure text")
        # Stream terminated cleanly with the failure event, not [DONE] success.
        if text.rstrip().endswith("data: [DONE]"):
            pytest.fail("failure stream ended with DONE")

    async def test_happy_path_still_completes_after_failure_guard_added(
        self,
    ) -> None:
        """Verify the happy path still completes after the failure guard."""
        st = self._seed_prev()

        async def ok_stream(
            *_args: object,
            **_kwargs: object,
        ) -> AsyncIterator[dict[str, object]]:
            await asyncio.sleep(0)
            yield {"type": "text_delta", "text": "hello"}
            yield {
                "type": "done",
                "result": TurnResult(
                    text="hello",
                    reasoning="",
                    image_urls=[],
                    response_id="r2",
                    conversation_id="conv-1",
                    parent_response_id="parent-1",
                    finish_reason="stop",
                ),
            }

        class FakeRequest:
            def __init__(self, body: dict[str, object]) -> None:
                self._body: dict[str, object] = body
                self.headers: dict[str, str] = {}

            async def json(self) -> dict[str, object]:
                return self._body

        req = FakeRequest(
            {"input": "follow up", "previous_response_id": "resp_prev", "stream": True},
        )
        with (
            patch.object(server, "stream_session_turn", new=ok_stream),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
            patch.object(server, "host_images", new=AsyncMock(return_value=[])),
        ):
            _resp, text = await self._collect(req)

        if "response.output_text.delta" not in text:
            pytest.fail("missing output delta")
        if "response.completed" not in text:
            pytest.fail("missing completed event")
        if "response.failed" in text:
            pytest.fail("happy path should not fail")
        if st is None:
            pytest.fail("seeded state missing")


if __name__ == "__main__":
    unittest.main()
