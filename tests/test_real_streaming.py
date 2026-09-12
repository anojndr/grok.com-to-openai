# Copyright (c) 2026 grok-to-openai-api contributors.
"""Tests for real real-time streaming in grok-to-openai-api."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from typing import TYPE_CHECKING, override
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse

import server
from accounts import Account
from grok_gateway import RenderFilter, TurnResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

CHAT_FIRST_CHUNK_DEADLINE = 0.15
CHAT_LAST_CHUNK_MIN = 0.20
RESP_FIRST_CHUNK_DEADLINE = 0.15
RESP_LAST_CHUNK_MIN = 0.15


class FakeRequest(Request):
    """Real Request carrying a canned JSON body."""

    def __init__(self, body: dict[str, object]) -> None:
        """Initialize the fake with a JSON-serializable body.

        Args:
            body: Request payload returned by ``json`` and ``body``.

        """
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
    async def json(self) -> dict[str, object]:
        return self._body_data

    @override
    async def body(self) -> bytes:
        return json.dumps(self._body_data).encode()


def _append_chat_deltas(
    chunk: str,
    now: float,
    start: float,
    collected: list[str],
    times: list[float],
) -> None:
    """Collect chat-completion deltas from one SSE chunk.

    Args:
        chunk: Raw SSE chunk text.
        now: Current monotonic timestamp for the chunk.
        start: Stream start timestamp.
        collected: Output list receiving delta text.
        times: Output list receiving per-delta elapsed times.

    """
    for line in chunk.split("\n"):
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload: object = json.loads(line[6:])
        if not isinstance(payload, dict):
            continue
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0]
        if not isinstance(first, dict):
            continue
        delta = first.get("delta", {})
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str) and content:
            collected.append(content)
            times.append(now - start)


def _append_response_deltas(
    chunk: str,
    now: float,
    start: float,
    collected: list[str],
    times: list[float],
) -> None:
    """Collect Responses-API deltas from one SSE chunk.

    Args:
        chunk: Raw SSE chunk text.
        now: Current monotonic timestamp for the chunk.
        start: Stream start timestamp.
        collected: Output list receiving delta text.
        times: Output list receiving per-delta elapsed times.

    """
    for block in chunk.split("\n\n"):
        for line in block.split("\n"):
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if not data or data == "[DONE]":
                continue
            payload: object = json.loads(data)
            if not isinstance(payload, dict):
                continue
            if payload.get("type") != "response.output_text.delta":
                continue
            delta = payload.get("delta")
            if isinstance(delta, str):
                collected.append(delta)
                times.append(now - start)


class RealStreamingTest(unittest.IsolatedAsyncioTestCase):
    """Verify streaming yields deltas in real time without buffering."""

    @override
    def setUp(self) -> None:
        """Clear server sessions before each test."""
        server.SESSIONS.clear()

    @override
    def tearDown(self) -> None:
        """Clear server sessions after each test."""
        server.SESSIONS.clear()

    @staticmethod
    async def test_chat_completions_real_streaming_timings() -> None:
        """Verify tokens arrive without buffering the whole response.

        Streams canned deltas through the chat-completions endpoint and
        checks that first and last chunk timings reflect real-time yield.
        """
        deltas = ["Hello", " world", " from", " real", " stream!"]
        delays = [0.05, 0.05, 0.05, 0.05, 0.05]

        fake_acc = Account(
            index=1,
            cookies={"sso": "tok", "x-userid": "uid-1"},
            user_id="uid-1",
        )
        fake_state = server.SessionState(
            account_key=fake_acc.key,
            grok=AsyncMock(),
        )

        async def fake_stream_turn(
            *_args: object,
            **_kwargs: object,
        ) -> AsyncIterator[dict[str, object]]:
            """Yield canned text deltas with delays.

            Yields:
                Stream event dicts with text deltas and a final done event.

            """
            for text, delay in zip(deltas, delays, strict=True):
                await asyncio.sleep(delay)
                yield {
                    "type": "text_delta",
                    "text": text,
                    "acc": fake_acc,
                    "state": fake_state,
                }
            yield {
                "type": "done",
                "result": TurnResult(text="Hello world from real stream!"),
                "acc": fake_acc,
                "state": fake_state,
            }

        body: dict[str, object] = {
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
            if not isinstance(resp, StreamingResponse):
                pytest.fail("expected streaming chat response")

            chunk_times: list[float] = []
            collected_deltas: list[str] = []
            async for chunk in resp.body_iterator:
                now = time.time()
                text = chunk if isinstance(chunk, str) else bytes(chunk).decode()
                _append_chat_deltas(
                    text,
                    now,
                    start_time,
                    collected_deltas,
                    chunk_times,
                )

        if collected_deltas != deltas:
            pytest.fail("chat deltas mismatch")
        if len(chunk_times) != len(deltas):
            pytest.fail("chat chunk count mismatch")
        # First chunk should have arrived well before the last chunk.
        if not chunk_times[0] < chunk_times[-1]:
            pytest.fail("first chunk not before last chunk")
        if not chunk_times[0] < CHAT_FIRST_CHUNK_DEADLINE:
            pytest.fail("first chunk arrived too late")
        if not chunk_times[-1] >= CHAT_LAST_CHUNK_MIN:
            pytest.fail("last chunk arrived too early")

    @staticmethod
    async def test_responses_api_real_streaming_timings() -> None:
        """Verify Responses SSE stream yields text deltas in real time."""
        deltas = ["Responding", " in", " real", " time!"]
        delays = [0.05, 0.05, 0.05, 0.05]

        fake_acc = Account(
            index=1,
            cookies={"sso": "tok", "x-userid": "uid-1"},
            user_id="uid-1",
        )
        fake_state = server.SessionState(
            account_key=fake_acc.key,
            grok=AsyncMock(),
        )

        async def fake_stream_turn(
            *_args: object,
            **_kwargs: object,
        ) -> AsyncIterator[dict[str, object]]:
            """Yield canned response deltas with delays.

            Yields:
                Stream event dicts with text deltas and a final done event.

            """
            for text, delay in zip(deltas, delays, strict=True):
                await asyncio.sleep(delay)
                yield {
                    "type": "text_delta",
                    "text": text,
                    "acc": fake_acc,
                    "state": fake_state,
                }
            yield {
                "type": "done",
                "result": TurnResult(text="Responding in real time!"),
                "acc": fake_acc,
                "state": fake_state,
            }

        body: dict[str, object] = {
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
            if not isinstance(resp, StreamingResponse):
                pytest.fail("expected streaming responses response")

            chunk_times: list[float] = []
            collected_deltas: list[str] = []
            async for chunk in resp.body_iterator:
                now = time.time()
                text = chunk if isinstance(chunk, str) else bytes(chunk).decode()
                _append_response_deltas(
                    text,
                    now,
                    start_time,
                    collected_deltas,
                    chunk_times,
                )

        if collected_deltas != deltas:
            pytest.fail("responses deltas mismatch")
        if len(chunk_times) != len(deltas):
            pytest.fail("responses chunk count mismatch")
        if not chunk_times[0] < chunk_times[-1]:
            pytest.fail("first chunk not before last chunk")
        if not chunk_times[0] < RESP_FIRST_CHUNK_DEADLINE:
            pytest.fail("first chunk arrived too late")
        if not chunk_times[-1] >= RESP_LAST_CHUNK_MIN:
            pytest.fail("last chunk arrived too early")

    @staticmethod
    async def test_render_filter_filters_in_stream() -> None:
        """Verify render cards filter without breaking streaming deltas."""
        filt = RenderFilter()
        chunks = [
            "Alpha ",
            "<grok:render",
            ' id="test">ignored</grok:render>',
            " Beta",
        ]
        out: list[str] = []
        for part in chunks:
            processed = filt.process(part)
            if processed:
                out.append(processed)
        flushed = filt.flush()
        if flushed:
            out.append(flushed)

        if "".join(out) != "Alpha  Beta":
            pytest.fail("render filter output mismatch")


if __name__ == "__main__":
    unittest.main()
