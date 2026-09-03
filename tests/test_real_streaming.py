"""Tests for real real-time streaming in grok-to-openai-api."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from typing import Any, override
from unittest.mock import AsyncMock, patch

from fastapi import Request

import server
from accounts import Account, AccountPool
from grok_gateway import GrokSession, TurnResult, RenderFilter


class FakeRequest(Request):
    """Real Request carrying a canned JSON body."""

    def __init__(self, body: dict[str, Any]) -> None:
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [],
            "query_string": b"",
            "server": ("test", 80),
            "scheme": "http",
            "client": ("test", 50000),
        }
        super().__init__(scope)
        self._body_data = body

    @override
    async def json(self) -> Any:
        return self._body_data

    @override
    async def body(self) -> bytes:
        return json.dumps(self._body_data).encode()


class RealStreamingTest(unittest.IsolatedAsyncioTestCase):
    @override
    def setUp(self):
        server.SESSIONS.clear()

    @override
    def tearDown(self):
        server.SESSIONS.clear()

    async def test_chat_completions_real_streaming_timings(self):
        """Verify tokens are yielded as they arrive without buffering the whole response."""
        deltas = ["Hello", " world", " from", " real", " stream!"]
        delays = [0.05, 0.05, 0.05, 0.05, 0.05]

        fake_acc = Account(
            index=1, cookies={"sso": "tok", "x-userid": "uid-1"}, user_id="uid-1"
        )
        fake_state = server.SessionState(account_key=fake_acc.key, grok=AsyncMock())

        async def fake_stream_turn(*args, **kwargs):
            for d, delay in zip(deltas, delays):
                await asyncio.sleep(delay)
                yield {
                    "type": "text_delta",
                    "text": d,
                    "acc": fake_acc,
                    "state": fake_state,
                }
            yield {
                "type": "done",
                "result": TurnResult(text="Hello world from real stream!"),
                "acc": fake_acc,
                "state": fake_state,
            }

        body = {
            "model": "grok-fast",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }

        with (
            patch("server.pick_account_and_stream_turn", fake_stream_turn),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
        ):
            start_time = time.time()
            resp = await server.chat_completions(FakeRequest(body))

            chunk_times = []
            collected_deltas = []
            async for chunk in resp.body_iterator:
                t = time.time()
                for line in chunk.split("\n"):
                    if line.startswith("data: ") and line != "data: [DONE]":
                        payload = json.loads(line[6:])
                        if payload.get("choices") and payload["choices"][0].get(
                            "delta", {}
                        ).get("content"):
                            collected_deltas.append(
                                payload["choices"][0]["delta"]["content"]
                            )
                            chunk_times.append(t - start_time)

        self.assertEqual(collected_deltas, deltas)
        self.assertEqual(len(chunk_times), len(deltas))
        # First chunk should have arrived well before the last chunk
        self.assertLess(chunk_times[0], chunk_times[-1])
        # First chunk elapsed time should be approximately ~0.05s (well under total duration ~0.25s)
        self.assertLess(chunk_times[0], 0.15)
        self.assertGreaterEqual(chunk_times[-1], 0.20)

    async def test_responses_api_real_streaming_timings(self):
        """Verify /v1/responses SSE stream yields text deltas in real-time."""
        deltas = ["Responding", " in", " real", " time!"]
        delays = [0.05, 0.05, 0.05, 0.05]

        fake_acc = Account(
            index=1, cookies={"sso": "tok", "x-userid": "uid-1"}, user_id="uid-1"
        )
        fake_state = server.SessionState(account_key=fake_acc.key, grok=AsyncMock())

        async def fake_stream_turn(*args, **kwargs):
            for d, delay in zip(deltas, delays):
                await asyncio.sleep(delay)
                yield {
                    "type": "text_delta",
                    "text": d,
                    "acc": fake_acc,
                    "state": fake_state,
                }
            yield {
                "type": "done",
                "result": TurnResult(text="Responding in real time!"),
                "acc": fake_acc,
                "state": fake_state,
            }

        body = {
            "model": "grok-fast",
            "stream": True,
            "input": "test prompt",
        }

        with (
            patch("server.pick_account_and_stream_turn", fake_stream_turn),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
        ):
            start_time = time.time()
            resp = await server.responses_api(FakeRequest(body))

            chunk_times = []
            collected_deltas = []
            async for chunk in resp.body_iterator:
                t = time.time()
                for block in chunk.split("\n\n"):
                    for line in block.split("\n"):
                        if line.startswith("data: "):
                            try:
                                payload = json.loads(line[6:])
                                if payload.get("type") == "response.output_text.delta":
                                    collected_deltas.append(payload["delta"])
                                    chunk_times.append(t - start_time)
                            except Exception:
                                pass

        self.assertEqual(collected_deltas, deltas)
        self.assertEqual(len(chunk_times), len(deltas))
        self.assertLess(chunk_times[0], chunk_times[-1])
        self.assertLess(chunk_times[0], 0.15)
        self.assertGreaterEqual(chunk_times[-1], 0.15)

    async def test_render_filter_filters_in_stream(self):
        """Verify <grok:render> cards are filtered out without breaking streaming deltas."""
        rf = RenderFilter()
        chunks = ["Alpha ", "<grok:render", ' id="test">ignored</grok:render>', " Beta"]
        out = []
        for c in chunks:
            processed = rf.process(c)
            if processed:
                out.append(processed)
        flushed = rf.flush()
        if flushed:
            out.append(flushed)

        self.assertEqual("".join(out), "Alpha  Beta")


if __name__ == "__main__":
    unittest.main()
