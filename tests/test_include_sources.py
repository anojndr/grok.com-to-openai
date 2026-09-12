# Copyright (c) 2026 grok-to-openai-api contributors.
"""Tests for the optional "Show Sources" bridge (include_sources).

Pins the llmcord-go-compatible appendix contract: enabled only via
`include_sources: true` (or G2O_INCLUDE_SOURCES=1), appended after the
answer content, with the stored session history kept clean.
Run: python3 -m unittest -v tests.test_include_sources
"""

from __future__ import annotations

import importlib
import json
import os
import re
import unittest
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast, override
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse
from websockets.protocol import State

import config
import server
from accounts import Account
from grok_gateway import GrokSession, TurnResult, extract_web_results

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


SOURCES = [
    {"url": "https://example.com/news", "title": "Example News"},
    {"url": "https://x.com/agency/status/1", "title": "Agency post"},
]

APPENDIX = (
    "\n\nSources\n"
    "1. [Example News](https://example.com/news) (example.com) "
    "via `latest news philippines`\n"
    "2. [Agency post](https://x.com/agency/status/1) (x.com) "
    "via `latest news philippines`\n"
    "\nSearch Queries\n"
    "1. `latest news philippines`"
)

_APPENDIX_ENTRY_CAP = 50


class FakeRequest(Request):
    """Minimal Request subclass for the endpoint handlers."""

    def __init__(self, body: dict[str, Any]) -> None:
        """Store the canned JSON body."""
        super().__init__(
            {
                "type": "http",
                "method": "POST",
                "headers": [],
                "query_string": b"",
                "server": ("testserver", 80),
                "scheme": "http",
                "path": "/",
                "root_path": "",
            },
        )
        self._fake_body: dict[str, Any] = body

    @override
    async def json(self) -> Any:
        return self._fake_body


def _search_frames() -> list[dict[str, Any]]:
    """Build the canned gateway frames for the ask test.

    Returns:
        The canned gateway event frames.

    """
    return [
        {
            "event": {
                "type": "conversation.item.added",
                "item": {"role": "user", "id": "umsg"},
            },
        },
        {
            "event": {
                "type": "response.chunk",
                "chunk": {
                    "tool_usage_card": {
                        "web_search": {
                            "args": {
                                "query": "latest news philippines",
                            },
                        },
                    },
                },
            },
        },
        {
            "event": {"type": "response.search.result"},
            "result": {
                "web_results": [
                    {"url": "https://example.com/news", "title": "Example News"},
                    {
                        "url": "https://x.com/agency/status/1",
                        "title": "Agency post",
                    },
                ],
            },
        },
        {
            "event": {
                "type": "response.output_text.delta",
                "delta": "Here is the news.",
            },
        },
        {
            "event": {
                "type": "response.done",
                "response": {"id": "resp-1", "status": "completed"},
            },
        },
    ]


class _MockWS:
    """Replay canned gateway frames without a live websocket."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        """Store the frames to replay."""
        self.protocol: Any = SimpleNamespace(state=State.OPEN)
        self._frames: list[dict[str, Any]] = list(frames)

    async def recv(self) -> str:
        """Return the next canned frame.

        Returns:
            The next canned frame as JSON text.

        Raises:
            TimeoutError: When no frames remain.

        """
        if not self._frames:
            raise TimeoutError
        return json.dumps(self._frames.pop(0))

    @staticmethod
    async def send(_message: object) -> None:
        """Ignore outbound payloads."""

    @staticmethod
    async def close() -> None:
        """Ignore close requests."""


async def _collect_done(sess: GrokSession, prompt: str) -> dict[str, Any]:
    """Drive one ask turn and return its done event.

    Returns:
        The done event of the driven turn.

    """
    done_event: dict[str, Any] | None = None
    async for ev in sess.ask(prompt, user_text=prompt):
        if ev.get("type") == "done":
            done_event = ev
    if done_event is None:
        pytest.fail("done event missing")
    return done_event


class SourceAppendixTest(unittest.TestCase):
    """Cover the source appendix formatting contract."""

    @staticmethod
    def test_full_form() -> None:
        """Verify the full appendix form."""
        appendix = server.source_appendix(SOURCES, "latest news philippines")
        if appendix != APPENDIX:
            pytest.fail("appendix mismatch")

    @staticmethod
    def test_empty_sources_yields_empty() -> None:
        """Verify empty sources yield an empty appendix."""
        if server.source_appendix([], "q"):
            pytest.fail("expected empty appendix")

    @staticmethod
    def test_no_query_omits_search_queries_and_via() -> None:
        """Verify an empty query omits attributions and query list."""
        out = server.source_appendix(SOURCES, "")
        if "via `" in out:
            pytest.fail("via attribution leaked without query")
        if "Search Queries" in out:
            pytest.fail("query list leaked without query")

    @staticmethod
    def test_title_falls_back_to_url_without_host_suffix() -> None:
        """Verify a missing title falls back to the bare URL."""
        out = server.source_appendix([{"url": "https://x.io/a", "title": ""}], "q")
        if "1. [https://x.io/a](https://x.io/a)" not in out:
            pytest.fail("title fallback missing")
        if ") (" in out:
            pytest.fail("host suffix leaked without title")

    @staticmethod
    def test_url_parens_and_spaces_escaped() -> None:
        """Verify parentheses and spaces are escaped in URLs."""
        out = server.source_appendix(
            [{"url": "https://x.io/a b)c", "title": "T"}],
            "q",
        )
        if "https://x.io/a%20b%29c" not in out:
            pytest.fail("URL escaping missing")

    @staticmethod
    def test_query_backticks_sanitized() -> None:
        """Verify backticks are sanitized from the query."""
        out = server.source_appendix(SOURCES[:1], "what's `up`")
        if "via `what's 'up'`" not in out:
            pytest.fail("query sanitizing missing")

    @staticmethod
    def test_multi_line_query_collapses_without_breaking_spans() -> None:
        """Verify multiline queries collapse without breaking spans."""
        out = server.source_appendix(SOURCES[:1], "latest news\nphilippines\t(2026)")
        expected = (
            "\n\nSources\n"
            "1. [Example News](https://example.com/news) (example.com) "
            "via `latest news philippines (2026)`\n"
            "\nSearch Queries\n"
            "1. `latest news philippines (2026)`"
        )
        if out != expected:
            pytest.fail("multiline query collapse mismatch")

    @staticmethod
    def test_title_newlines_collapsed() -> None:
        """Verify newlines collapse in titles."""
        out = server.source_appendix(
            [{"url": "https://x.io/a", "title": "line1\nline2"}],
            "q",
        )
        if "[line1 line2](https://x.io/a)" not in out:
            pytest.fail("title newline collapse missing")

    @staticmethod
    def test_caps_at_50_entries() -> None:
        """Verify the appendix caps at the configured entry limit."""
        if server.SOURCE_APPENDIX_MAX != _APPENDIX_ENTRY_CAP:
            pytest.fail("appendix cap constant drifted")
        many = [{"url": f"https://x.io/{i}", "title": f"t{i}"} for i in range(60)]
        out = server.source_appendix(many, "q")
        entries = [line for line in out.splitlines() if re.match(r"^\d+\. \[", line)]
        if len(entries) != _APPENDIX_ENTRY_CAP:
            pytest.fail("appendix entry cap mismatch")


class IncludeSourcesFlagTest(unittest.TestCase):
    """Cover the include-sources flag resolution."""

    @staticmethod
    def test_flag_none_uses_config_default() -> None:
        """Verify an unset flag falls back to the config default."""
        with patch("server.INCLUDE_SOURCES", new=True):
            if not server.include_sources(flag=None):
                pytest.fail("config default True not honored")
        with patch("server.INCLUDE_SOURCES", new=False):
            if server.include_sources(flag=None):
                pytest.fail("config default False not honored")

    @staticmethod
    def test_flag_overrides_config_default() -> None:
        """Verify an explicit flag overrides the config default."""
        with patch("server.INCLUDE_SOURCES", new=True):
            if server.include_sources(flag=False):
                pytest.fail("explicit False did not override True default")
        with patch("server.INCLUDE_SOURCES", new=False):
            if not server.include_sources(flag=True):
                pytest.fail("explicit True did not override False default")

    @staticmethod
    def test_string_flags_parse_like_env_values() -> None:
        """Verify string flags parse like environment values."""
        for truthy in ("1", "true", "TRUE", "yes", "on", " on "):
            if not server.include_sources(flag=truthy):
                pytest.fail(f"truthy flag not honored: {truthy!r}")
        for falsy in ("0", "false", "no", "off", "", "garbage"):
            if server.include_sources(flag=falsy):
                pytest.fail(f"falsy flag not honored: {falsy!r}")

    @staticmethod
    def test_env_var_grok_include_sources_fallback() -> None:
        """Verify the include-sources environment fallback chain."""
        with patch.dict(os.environ, {"GROK_INCLUDE_SOURCES": "1"}, clear=False):
            os.environ.pop("G2O_INCLUDE_SOURCES", None)
            importlib.reload(config)
            if not config.INCLUDE_SOURCES:
                pytest.fail("GROK_INCLUDE_SOURCES fallback not honored")

        with patch.dict(
            os.environ,
            {"G2O_INCLUDE_SOURCES": "0", "GROK_INCLUDE_SOURCES": "1"},
            clear=False,
        ):
            importlib.reload(config)
            if config.INCLUDE_SOURCES:
                pytest.fail("G2O_INCLUDE_SOURCES did not take precedence")

        # Restore default state
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("G2O_INCLUDE_SOURCES", None)
            os.environ.pop("GROK_INCLUDE_SOURCES", None)
            importlib.reload(config)
            importlib.reload(server)


class WebSourcesExtractionTest(unittest.TestCase):
    """Cover web result extraction across event shapes."""

    EVENT_SEARCH: ClassVar[dict[str, Any]] = {
        "event": {"type": "response.search.result"},
        "result": {
            "search_type": "web_search",
            "web_results": [{"url": "https://a.io/1", "title": "A"}],
        },
    }

    EVENT_TOOL: ClassVar[dict[str, Any]] = {
        "event": {"type": "response.grok.output"},
        "output": {
            "tool_result": {
                "web_search_results": [{"url": "https://b.io/2", "title": " B "}],
            },
        },
    }

    EVENT_CHUNK: ClassVar[dict[str, Any]] = {
        "event": {"type": "response.chunk"},
        "chunk": {
            "tool_result": {
                "web_search_results": [{"url": "https://c.io/3", "title": "C"}],
            },
        },
    }

    EVENT_CHUNK_LIVE_WEBPAGES: ClassVar[dict[str, Any]] = {
        "event": {"type": "response.chunk"},
        "chunk": {
            "tool_result": {
                "tool_call_id": "call-1",
                "web_search": {
                    "webpages": [
                        {
                            "url": "https://live.news/article",
                            "title": "Live News",
                            "snippet": "Snippet text",
                        },
                    ],
                },
            },
        },
    }

    def test_search_result_event_extracts_entries(self) -> None:
        """Verify search result events yield entries."""
        got = extract_web_results(self.EVENT_SEARCH)
        if got != [{"url": "https://a.io/1", "title": "A"}]:
            pytest.fail("search result entries mismatch")

    def test_tool_result_output_event_extracts_entries(self) -> None:
        """Verify tool result output events yield entries."""
        got = extract_web_results(self.EVENT_TOOL)
        if got != [{"url": "https://b.io/2", "title": "B"}]:
            pytest.fail("tool output entries mismatch")

    def test_tool_result_chunk_event_extracts_entries(self) -> None:
        """Verify tool result chunk events yield entries."""
        got = extract_web_results(self.EVENT_CHUNK)
        if got != [{"url": "https://c.io/3", "title": "C"}]:
            pytest.fail("tool chunk entries mismatch")

    def test_tool_result_live_webpages_shape_extracts_entries(self) -> None:
        """Verify live webpage chunks yield entries."""
        got = extract_web_results(self.EVENT_CHUNK_LIVE_WEBPAGES)
        if got != [{"url": "https://live.news/article", "title": "Live News"}]:
            pytest.fail("live webpages entries mismatch")

    @staticmethod
    def test_title_falls_back_to_url() -> None:
        """Verify a missing title falls back to the URL."""
        ev = {"result": {"web_results": [{"url": "https://d.io/4"}]}}
        got = extract_web_results(ev)
        if got != [{"url": "https://d.io/4", "title": "https://d.io/4"}]:
            pytest.fail("title fallback mismatch")

    @staticmethod
    def test_malformed_entries_ignored() -> None:
        """Verify malformed entries are ignored."""
        ev = {
            "result": {
                "web_results": [
                    None,
                    {"url": " "},
                    {"url": "https://e.io/5", "title": 3},
                ],
            },
        }
        got = extract_web_results(ev)
        if got != [{"url": "https://e.io/5", "title": "3"}]:
            pytest.fail("malformed entries not ignored")

    @staticmethod
    def test_unrelated_event_yields_nothing() -> None:
        """Verify unrelated events yield nothing."""
        if extract_web_results({"event": {"type": "response.done"}}) != []:
            pytest.fail("unrelated event yielded entries")


class GatewaySessionAskSourcesTest(unittest.IsolatedAsyncioTestCase):
    """Exercise source collection through session ask."""

    @staticmethod
    async def test_ask_collects_sources_and_search_queries_into_turn_result() -> None:
        """Verify ask collects sources and queries into the result."""
        sess = GrokSession("ck", "uid")
        sess.conversation_id = "conv-1"
        sess.ws = cast("ClientConnection", _MockWS(_search_frames()))

        done = await _collect_done(sess, "latest news philippines")
        turn = done["result"]
        if not isinstance(turn, TurnResult):
            pytest.fail("done result is not a TurnResult")
        if turn.text != "Here is the news.":
            pytest.fail("turn text mismatch")
        if turn.sources != SOURCES:
            pytest.fail("turn sources mismatch")
        if turn.search_queries != ["latest news philippines"]:
            pytest.fail("search queries mismatch")


class ChatCompletionsAppendixEndpointTest(unittest.IsolatedAsyncioTestCase):
    """Cover appendix bridging in chat completions."""

    @staticmethod
    async def test_chat_non_stream_appends_sources_when_requested() -> None:
        """Verify non-stream chat appends sources when requested."""
        body = {
            "model": "grok-fast",
            "include_sources": True,
            "messages": [{"role": "user", "content": "latest news philippines"}],
        }
        fake_result = TurnResult(
            text="Here is the news.",
            sources=SOURCES,
            search_queries=["latest news philippines"],
        )
        fake_acc = Account(
            index=1,
            cookies={"sso": "tok", "x-userid": "uid-1"},
            user_id="uid-1",
        )
        fake_state = server.SessionState(account_key=fake_acc.key, grok=AsyncMock())

        with (
            patch(
                "server.pick_account_and_turn",
                new=AsyncMock(return_value=(fake_acc, fake_result, [], fake_state)),
            ),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
        ):
            resp = await server.chat_completions(FakeRequest(body))

        data = json.loads(bytes(resp.body))
        content = data["choices"][0]["message"]["content"]
        if content != "Here is the news." + APPENDIX:
            pytest.fail("sources appendix missing from chat reply")

    @staticmethod
    async def test_chat_non_stream_omits_sources_by_default() -> None:
        """Verify non-stream chat omits sources by default."""
        body = {
            "model": "grok-fast",
            "messages": [{"role": "user", "content": "latest news philippines"}],
        }
        fake_result = TurnResult(
            text="Here is the news.",
            sources=SOURCES,
            search_queries=["latest news philippines"],
        )
        fake_acc = Account(
            index=1,
            cookies={"sso": "tok", "x-userid": "uid-1"},
            user_id="uid-1",
        )
        fake_state = server.SessionState(account_key=fake_acc.key, grok=AsyncMock())

        with (
            patch(
                "server.pick_account_and_turn",
                new=AsyncMock(return_value=(fake_acc, fake_result, [], fake_state)),
            ),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
        ):
            resp = await server.chat_completions(FakeRequest(body))

        data = json.loads(bytes(resp.body))
        content = data["choices"][0]["message"]["content"]
        if content != "Here is the news.":
            pytest.fail("sources appendix leaked into chat reply")

    @staticmethod
    async def test_chat_stream_appends_sources_chunk() -> None:
        """Verify streamed chat appends sources to the chunks."""
        body = {
            "model": "grok-fast",
            "stream": True,
            "include_sources": True,
            "messages": [{"role": "user", "content": "latest news philippines"}],
        }
        fake_result = TurnResult(
            text="Here is the news.",
            sources=SOURCES,
            search_queries=["latest news philippines"],
        )
        fake_acc = Account(
            index=1,
            cookies={"sso": "tok", "x-userid": "uid-1"},
            user_id="uid-1",
        )
        fake_state = server.SessionState(account_key=fake_acc.key, grok=AsyncMock())

        with (
            patch(
                "server.pick_account_and_turn",
                new=AsyncMock(return_value=(fake_acc, fake_result, [], fake_state)),
            ),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
        ):
            resp = await server.chat_completions(FakeRequest(body))
            if not isinstance(resp, StreamingResponse):
                pytest.fail("expected streaming chat response")
            chunks: list[str] = [
                c if isinstance(c, str) else bytes(c).decode("utf-8")
                async for c in resp.body_iterator
            ]

        contents = []
        for chunk in chunks:
            line = chunk.strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line[len("data: ") :])
            if payload.get("object") != "chat.completion.chunk":
                continue
            delta = payload["choices"][0]["delta"]
            if "content" in delta:
                contents.append(delta["content"])

        if "".join(contents) != "Here is the news." + APPENDIX:
            pytest.fail("sources appendix missing from chat stream")


class ResponsesApiAppendixEndpointTest(unittest.IsolatedAsyncioTestCase):
    """Cover appendix bridging in the responses API."""

    @staticmethod
    async def test_responses_non_stream_appends_sources_when_requested() -> None:
        """Verify non-stream responses append sources when requested."""
        body = {
            "model": "grok-fast",
            "include_sources": True,
            "input": "latest news philippines",
        }
        fake_result = TurnResult(
            text="Here is the news.",
            sources=SOURCES,
            search_queries=["latest news philippines"],
        )
        fake_acc = Account(
            index=1,
            cookies={"sso": "tok", "x-userid": "uid-1"},
            user_id="uid-1",
        )
        fake_state = server.SessionState(account_key=fake_acc.key, grok=AsyncMock())

        with (
            patch(
                "server.pick_account_and_turn",
                new=AsyncMock(return_value=(fake_acc, fake_result, [], fake_state)),
            ),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
        ):
            resp = await server.responses_api(FakeRequest(body))

        data = json.loads(bytes(resp.body))
        item_text = data["output"][0]["content"][0]["text"]
        if item_text != "Here is the news." + APPENDIX:
            pytest.fail("sources appendix missing from output item")
        if data["output_text"] != "Here is the news." + APPENDIX:
            pytest.fail("sources appendix missing from output text")

    @staticmethod
    async def test_responses_stream_appends_sources_in_events() -> None:
        """Verify streamed responses carry sources in events."""
        body = {
            "model": "grok-fast",
            "stream": True,
            "include_sources": True,
            "input": "latest news philippines",
        }
        fake_result = TurnResult(
            text="Here is the news.",
            sources=SOURCES,
            search_queries=["latest news philippines"],
        )
        fake_acc = Account(
            index=1,
            cookies={"sso": "tok", "x-userid": "uid-1"},
            user_id="uid-1",
        )
        fake_state = server.SessionState(account_key=fake_acc.key, grok=AsyncMock())

        with (
            patch(
                "server.pick_account_and_turn",
                new=AsyncMock(return_value=(fake_acc, fake_result, [], fake_state)),
            ),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
        ):
            resp = await server.responses_api(FakeRequest(body))
            if not isinstance(resp, StreamingResponse):
                pytest.fail("expected streaming responses response")
            events = {}
            for chunk in [
                c if isinstance(c, str) else bytes(c).decode("utf-8")
                async for c in resp.body_iterator
            ]:
                for block in chunk.split("\n\n"):
                    data = None
                    for line in block.splitlines():
                        if line.startswith("data: "):
                            data = line[len("data: ") :]
                    if data is None or '"type"' not in data:
                        continue
                    payload = json.loads(data)
                    events.setdefault(payload["type"], []).append(payload)

        deltas = "".join(p["delta"] for p in events["response.output_text.delta"])
        if deltas != "Here is the news." + APPENDIX:
            pytest.fail("sources appendix missing from deltas")
        done = events["response.output_item.done"][0]
        completed = events["response.completed"][0]
        done_text = done["item"]["content"][0]["text"]
        if done_text != "Here is the news." + APPENDIX:
            pytest.fail("sources appendix missing from done item")
        completed_text = completed["response"]["output"][0]["content"][0]["text"]
        if completed_text != "Here is the news." + APPENDIX:
            pytest.fail("sources appendix missing from completed response")


if __name__ == "__main__":
    unittest.main()
