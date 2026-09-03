"""Regression tests: attachment uploads must never be silently dropped.

2026-08-25 incident ("fact check this" + image): the statsig seed/hex pair
could not bootstrap (grok.com HTML served a Cloudflare challenge), so
statsig.generate raised before any HTTP call, stream_session_turn swallowed
the error into attachment_ids=None, and grok confidently answered as if no
image had been attached.
"""

from __future__ import annotations

import base64
import unittest
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, override
from unittest.mock import AsyncMock, patch

import server
from accounts import Account
from grok_gateway import GatewayError, GrokSession, TurnResult
from statsig import StatsigGenerator
from uploads import UploadError, _sig_headers

SEED_B64 = base64.b64encode(bytes(range(48))).decode()


class SigHeaderTests(unittest.TestCase):
    def test_omitted_when_generator_not_ready(self):
        gen = StatsigGenerator()
        self.assertFalse(gen.ready)
        self.assertEqual(_sig_headers(gen, "/rest/app-chat/x", "POST"), {})
        self.assertEqual(_sig_headers(None, "/rest/app-chat/x", "POST"), {})

    def test_included_when_ready(self):
        gen = StatsigGenerator()
        gen.set_pair(SEED_B64, "deadbeef")
        header = _sig_headers(gen, "/rest/app-chat/x", "POST")
        self.assertIn("x-statsig-id", header)
        self.assertTrue(header["x-statsig-id"])


class FakeSess(GrokSession):
    def __init__(self) -> None:
        super().__init__("", "", "fast")
        self.cookie_header = "ck"
        self.last_kwargs: dict[str, Any] | None = None

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
    ) -> AsyncIterator[dict[str, Any]]:
        self.last_kwargs = {
            "attachment_ids": attachment_ids,
            "system_prompt": system_prompt,
        }
        yield {"type": "done", "result": SimpleNamespace(text="ok")}


class StreamTurnUploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_raises_gateway_error_when_every_upload_fails(self):
        sess = FakeSess()
        jobs = [{"name": "image.png", "data": b"xx", "mime": "image/png"}]
        with (
            patch.object(
                server,
                "upload_file",
                new=AsyncMock(side_effect=UploadError("init failed 403")),
            ),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            with self.assertRaises(GatewayError) as ctx:
                async for _ in server.stream_session_turn(
                    sess, "fact check this", file_jobs=jobs
                ):
                    pass
        self.assertIn("attachment upload failed", str(ctx.exception))
        self.assertIsNone(sess.last_kwargs)

    async def test_partial_failure_still_sends_uploaded_files(self):
        sess = FakeSess()
        jobs = [
            {"name": "a.png", "data": b"a", "mime": "image/png"},
            {"name": "b.png", "data": b"b", "mime": "image/png"},
        ]
        up = AsyncMock(
            side_effect=[{"fileMetadataId": "fid-ok"}, UploadError("put failed 500")]
        )
        with (
            patch.object(server, "upload_file", new=up),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            events = [
                ev
                async for ev in server.stream_session_turn(
                    sess, "prompt", file_jobs=jobs
                )
            ]
        self.assertTrue(any(e["type"] == "done" for e in events))
        assert isinstance(sess.last_kwargs, dict)
        self.assertEqual(sess.last_kwargs["attachment_ids"], ["fid-ok"])

    async def test_no_files_passes_through_untouched(self):
        sess = FakeSess()
        with (
            patch.object(server, "upload_file", new=AsyncMock()) as up,
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            events = [ev async for ev in server.stream_session_turn(sess, "hello")]
        up.assert_not_awaited()
        self.assertTrue(any(e["type"] == "done" for e in events))

    async def test_preexisting_ids_do_not_rescue_failed_uploads(self):
        """A pass-through file_id must not mask a fully-failed upload batch."""
        sess = FakeSess()
        jobs = [
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
        ):
            with self.assertRaises(GatewayError):
                async for _ in server.stream_session_turn(
                    sess,
                    "fact check this",
                    attachment_ids=["caller-supplied"],
                    file_jobs=jobs,
                ):
                    pass
        self.assertIsNone(sess.last_kwargs)

    async def test_upload_without_metadata_id_counts_as_failure(self):
        """A silent no-fileMetadataId response is a dropped file, not success."""
        sess = FakeSess()
        jobs = [{"name": "image.png", "data": b"x", "mime": "image/png"}]
        with (
            patch.object(server, "upload_file", new=AsyncMock(return_value={})),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            with self.assertRaises(GatewayError) as ctx:
                async for _ in server.stream_session_turn(
                    sess, "prompt", file_jobs=jobs
                ):
                    pass
        self.assertIn("no fileMetadataId", str(ctx.exception))

    def test_sig_headers_survive_generate_crash(self):
        gen = StatsigGenerator()
        gen.set_pair(base64.b64encode(b"short").decode(), "hex")  # <48-byte seed
        self.assertTrue(gen.ready)
        with patch.object(
            gen,
            "generate",
            side_effect=RuntimeError("statsig seed must be at least 48 bytes"),
        ):
            self.assertEqual(_sig_headers(gen, "/p", "POST"), {})


class EnsurePairResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_page_keeps_generator_unready_without_raising(self):
        gen = StatsigGenerator()
        calls = []

        async def fetch_page():
            calls.append(1)
            return None  # fetch_page now returns None for challenge pages

        await gen.ensure_pair(fetch_page)
        self.assertFalse(gen.ready)
        self.assertEqual(len(calls), 1)


class ResponsesStreamFailureTests(unittest.IsolatedAsyncioTestCase):
    """A turn failure after initial SSE frames must yield response.failed,
    not abort the connection (review finding on responses_api sse())."""

    @override
    def setUp(self):
        import os
        import shutil
        import tempfile
        from accounts import AccountPool
        from session_store import SqliteStore

        self.tmp_dir = tempfile.mkdtemp()
        self.store = SqliteStore(os.path.join(self.tmp_dir, "t.db"))
        self._orig_store = server.store
        server.store = self.store
        self.fake_acc = Account(
            index=1, cookies={"sso": "tok", "x-userid": "uid-1"}, user_id="uid-1"
        )
        self._orig_pool = server.pool
        pool = AccountPool(os.path.join(self.tmp_dir, "accounts.txt"))
        pool._accounts = [self.fake_acc]
        server.pool = pool
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.addCleanup(setattr, server, "store", self._orig_store)
        self.addCleanup(setattr, server, "pool", self._orig_pool)
        self.addCleanup(server.SESSIONS.clear)

    def _seed_prev(self):
        fake_grok = FakeSess()
        fake_grok.conversation_id = "conv-1"
        fake_grok.last_parent_response_id = "parent-1"
        st = server.SessionState(
            account_key=self.fake_acc.key, grok=fake_grok, user_chain=["first question"]
        )
        server.SESSIONS["resp_prev"] = st
        return st

    async def _collect(self, req):
        resp = await server.responses_api(req)
        chunks = [c async for c in resp.body_iterator]
        return resp, "".join(chunks)

    async def test_gateway_error_streams_response_failed_event(self):
        self._seed_prev()

        async def failing_stream(*args, **kwargs):
            raise GatewayError(
                "upstream", "attachment upload failed: image: init failed 403"
            )
            yield  # pragma: no cover - makes this an async generator

        class FakeRequest:
            def __init__(self, body):
                self._body = body
                self.headers = {}

            async def json(self):
                return self._body

        req = FakeRequest(
            {"input": "follow up", "previous_response_id": "resp_prev", "stream": True}
        )
        with (
            patch.object(server, "stream_session_turn", new=failing_stream),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
            patch.object(server, "host_images", new=AsyncMock(return_value=[])),
        ):
            resp, text = await self._collect(req)

        self.assertIn("response.failed", text)
        self.assertIn("attachment upload failed", text)
        # Stream terminated cleanly with the failure event, not [DONE] success.
        self.assertFalse(text.rstrip().endswith("data: [DONE]"))

    async def test_happy_path_still_completes_after_failure_guard_added(self):
        st = self._seed_prev()

        async def ok_stream(*args, **kwargs):
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
            def __init__(self, body):
                self._body = body
                self.headers = {}

            async def json(self):
                return self._body

        req = FakeRequest(
            {"input": "follow up", "previous_response_id": "resp_prev", "stream": True}
        )
        with (
            patch.object(server, "stream_session_turn", new=ok_stream),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
            patch.object(server, "host_images", new=AsyncMock(return_value=[])),
        ):
            resp, text = await self._collect(req)

        self.assertIn("response.output_text.delta", text)
        self.assertIn("response.completed", text)
        self.assertNotIn("response.failed", text)
        self.assertIsNotNone(st)


if __name__ == "__main__":
    unittest.main()
