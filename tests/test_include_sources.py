"""Tests for the optional "Show Sources" bridge (include_sources).

Pins the llmcord-go-compatible appendix contract: enabled only via
`include_sources: true` (or G2O_INCLUDE_SOURCES=1), appended after the
answer content, with the stored session history kept clean.
Run: python3 -m unittest -v tests.test_include_sources
"""

from __future__ import annotations

import asyncio
import json
import re
import unittest
from collections.abc import AsyncIterable, Iterable
from types import SimpleNamespace
from typing import Any, Literal, overload, override
from unittest.mock import AsyncMock, patch

import server
from accounts import Account
from fastapi import Request
from grok_gateway import GrokSession, TurnResult, extract_web_results
from websockets.asyncio.client import ClientConnection
from websockets.frames import CloseCode
from websockets.protocol import State
from websockets.typing import Data, DataLike

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


class FakeRequest(Request):
    """Minimal Request subclass for the endpoint handlers."""

    def __init__(self, body: dict[str, Any]) -> None:
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
            }
        )
        self._fake_body: dict[str, Any] = body

    @override
    async def json(self) -> Any:
        return self._fake_body


class SourceAppendixTest(unittest.TestCase):
    def test_full_form(self):
        self.assertEqual(
            server._source_appendix(SOURCES, "latest news philippines"), APPENDIX
        )

    def test_empty_sources_yields_empty(self):
        self.assertEqual(server._source_appendix([], "q"), "")

    def test_no_query_omits_search_queries_and_via(self):
        out = server._source_appendix(SOURCES, "")
        self.assertNotIn("via `", out)
        self.assertNotIn("Search Queries", out)

    def test_title_falls_back_to_url_without_host_suffix(self):
        out = server._source_appendix([{"url": "https://x.io/a", "title": ""}], "q")
        self.assertIn("1. [https://x.io/a](https://x.io/a)", out)
        self.assertNotIn(") (", out)

    def test_url_parens_and_spaces_escaped(self):
        out = server._source_appendix(
            [{"url": "https://x.io/a b)c", "title": "T"}], "q"
        )
        self.assertIn("https://x.io/a%20b%29c", out)

    def test_query_backticks_sanitized(self):
        out = server._source_appendix(SOURCES[:1], "what's `up`")
        self.assertIn("via `what's 'up'`", out)

    def test_multi_line_query_collapses_without_breaking_spans(self):
        out = server._source_appendix(SOURCES[:1], "latest news\nphilippines\t(2026)")
        expected = (
            "\n\nSources\n"
            "1. [Example News](https://example.com/news) (example.com) "
            "via `latest news philippines (2026)`\n"
            "\nSearch Queries\n"
            "1. `latest news philippines (2026)`"
        )
        self.assertEqual(out, expected)

    def test_title_newlines_collapsed(self):
        out = server._source_appendix(
            [{"url": "https://x.io/a", "title": "line1\nline2"}], "q"
        )
        self.assertIn("[line1 line2](https://x.io/a)", out)

    def test_caps_at_50_entries(self):
        many = [{"url": f"https://x.io/{i}", "title": f"t{i}"} for i in range(60)]
        out = server._source_appendix(many, "q")
        entries = [l for l in out.splitlines() if re.match(r"^\d+\. \[", l)]
        self.assertEqual(len(entries), 50)


class IncludeSourcesFlagTest(unittest.TestCase):
    def test_flag_none_uses_config_default(self):
        with patch("server.INCLUDE_SOURCES", True):
            self.assertTrue(server._include_sources(None))
        with patch("server.INCLUDE_SOURCES", False):
            self.assertFalse(server._include_sources(None))

    def test_flag_overrides_config_default(self):
        with patch("server.INCLUDE_SOURCES", True):
            self.assertFalse(server._include_sources(False))
        with patch("server.INCLUDE_SOURCES", False):
            self.assertTrue(server._include_sources(True))

    def test_string_flags_parse_like_env_values(self):
        for truthy in ("1", "true", "TRUE", "yes", "on", " on "):
            self.assertTrue(server._include_sources(truthy))
        for falsy in ("0", "false", "no", "off", "", "garbage"):
            self.assertFalse(server._include_sources(falsy))

    def test_env_var_grok_include_sources_fallback(self):
        import importlib
        import os
        import config

        with patch.dict(os.environ, {"GROK_INCLUDE_SOURCES": "1"}, clear=False):
            os.environ.pop("G2O_INCLUDE_SOURCES", None)
            importlib.reload(config)
            self.assertTrue(config.INCLUDE_SOURCES)

        with patch.dict(
            os.environ,
            {"G2O_INCLUDE_SOURCES": "0", "GROK_INCLUDE_SOURCES": "1"},
            clear=False,
        ):
            importlib.reload(config)
            self.assertFalse(config.INCLUDE_SOURCES)

        # Restore default state
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("G2O_INCLUDE_SOURCES", None)
            os.environ.pop("GROK_INCLUDE_SOURCES", None)
            importlib.reload(config)
            importlib.reload(server)


class WebSourcesExtractionTest(unittest.TestCase):
    EVENT_SEARCH = {
        "event": {"type": "response.search.result"},
        "result": {
            "search_type": "web_search",
            "web_results": [{"url": "https://a.io/1", "title": "A"}],
        },
    }

    EVENT_TOOL = {
        "event": {"type": "response.grok.output"},
        "output": {
            "tool_result": {
                "web_search_results": [{"url": "https://b.io/2", "title": " B "}]
            }
        },
    }

    EVENT_CHUNK = {
        "event": {"type": "response.chunk"},
        "chunk": {
            "tool_result": {
                "web_search_results": [{"url": "https://c.io/3", "title": "C"}]
            }
        },
    }

    EVENT_CHUNK_LIVE_WEBPAGES = {
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
                        }
                    ]
                },
            }
        },
    }

    def test_search_result_event_extracts_entries(self):
        self.assertEqual(
            extract_web_results(self.EVENT_SEARCH),
            [{"url": "https://a.io/1", "title": "A"}],
        )

    def test_tool_result_output_event_extracts_entries(self):
        self.assertEqual(
            extract_web_results(self.EVENT_TOOL),
            [{"url": "https://b.io/2", "title": "B"}],
        )

    def test_tool_result_chunk_event_extracts_entries(self):
        self.assertEqual(
            extract_web_results(self.EVENT_CHUNK),
            [{"url": "https://c.io/3", "title": "C"}],
        )

    def test_tool_result_live_webpages_shape_extracts_entries(self):
        self.assertEqual(
            extract_web_results(self.EVENT_CHUNK_LIVE_WEBPAGES),
            [{"url": "https://live.news/article", "title": "Live News"}],
        )

    def test_title_falls_back_to_url(self):
        ev = {"result": {"web_results": [{"url": "https://d.io/4"}]}}
        self.assertEqual(
            extract_web_results(ev),
            [{"url": "https://d.io/4", "title": "https://d.io/4"}],
        )

    def test_malformed_entries_ignored(self):
        ev = {
            "result": {
                "web_results": [
                    None,
                    {"url": " "},
                    {"url": "https://e.io/5", "title": 3},
                ]
            }
        }
        self.assertEqual(
            extract_web_results(ev), [{"url": "https://e.io/5", "title": "3"}]
        )

    def test_unrelated_event_yields_nothing(self):
        self.assertEqual(extract_web_results({"event": {"type": "response.done"}}), [])


class GatewaySessionAskSourcesTest(unittest.IsolatedAsyncioTestCase):
    async def test_ask_collects_sources_and_search_queries_into_turn_result(self):
        frames: list[dict[str, Any]] = [
            {
                "event": {
                    "type": "conversation.item.added",
                    "item": {"role": "user", "id": "umsg"},
                }
            },
            {
                "event": {
                    "type": "response.chunk",
                    "chunk": {
                        "tool_usage_card": {
                            "web_search": {"args": {"query": "latest news philippines"}}
                        }
                    },
                }
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
                    ]
                },
            },
            {
                "event": {
                    "type": "response.output_text.delta",
                    "delta": "Here is the news.",
                }
            },
            {
                "event": {
                    "type": "response.done",
                    "response": {"id": "resp-1", "status": "completed"},
                }
            },
        ]

        class MockWS(ClientConnection):
            def __init__(self) -> None:
                self.protocol: Any = SimpleNamespace(state=State.OPEN)
                self._frames: list[dict[str, Any]] = list(frames)

            @overload
            async def recv(self, decode: Literal[True]) -> str: ...

            @overload
            async def recv(self, decode: Literal[False]) -> bytes: ...

            @overload
            async def recv(self, decode: bool | None = None) -> Data: ...

            @override
            async def recv(self, decode: bool | None = None) -> Data:
                if not self._frames:
                    raise asyncio.TimeoutError()
                return json.dumps(self._frames.pop(0))

            @override
            async def send(
                self,
                message: DataLike | Iterable[DataLike] | AsyncIterable[DataLike],
                *,
                text: bool | None = None,
            ) -> None:
                return None

            @override
            async def close(
                self, code: CloseCode | int = CloseCode.NORMAL_CLOSURE, reason: str = ""
            ) -> None:
                return None

        sess = GrokSession("ck", "uid")
        sess.conversation_id = "conv-1"
        sess.ws = MockWS()

        done_event: dict[str, Any] | None = None
        async for ev in sess.ask(
            "latest news philippines", user_text="latest news philippines"
        ):
            if ev.get("type") == "done":
                done_event = ev

        self.assertIsNotNone(done_event)
        assert isinstance(done_event, dict)
        turn = done_event["result"]
        assert isinstance(turn, TurnResult)
        result = turn
        self.assertEqual(result.text, "Here is the news.")
        self.assertEqual(result.sources, SOURCES)
        self.assertEqual(result.search_queries, ["latest news philippines"])


class ChatCompletionsAppendixEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_chat_non_stream_appends_sources_when_requested(self):
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
            index=1, cookies={"sso": "tok", "x-userid": "uid-1"}, user_id="uid-1"
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

        data = json.loads(resp.body)
        self.assertEqual(
            data["choices"][0]["message"]["content"], "Here is the news." + APPENDIX
        )

    async def test_chat_non_stream_omits_sources_by_default(self):
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
            index=1, cookies={"sso": "tok", "x-userid": "uid-1"}, user_id="uid-1"
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

        data = json.loads(resp.body)
        self.assertEqual(data["choices"][0]["message"]["content"], "Here is the news.")

    async def test_chat_stream_appends_sources_chunk(self):
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
            index=1, cookies={"sso": "tok", "x-userid": "uid-1"}, user_id="uid-1"
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
            chunks = [c async for c in resp.body_iterator]

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

        self.assertEqual("".join(contents), "Here is the news." + APPENDIX)


class ResponsesApiAppendixEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_responses_non_stream_appends_sources_when_requested(self):
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
            index=1, cookies={"sso": "tok", "x-userid": "uid-1"}, user_id="uid-1"
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

        data = json.loads(resp.body)
        self.assertEqual(
            data["output"][0]["content"][0]["text"], "Here is the news." + APPENDIX
        )
        self.assertEqual(data["output_text"], "Here is the news." + APPENDIX)

    async def test_responses_stream_appends_sources_in_events(self):
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
            index=1, cookies={"sso": "tok", "x-userid": "uid-1"}, user_id="uid-1"
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
            events = {}
            for chunk in [c async for c in resp.body_iterator]:
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
        self.assertEqual(deltas, "Here is the news." + APPENDIX)
        done = events["response.output_item.done"][0]
        completed = events["response.completed"][0]
        self.assertEqual(
            done["item"]["content"][0]["text"], "Here is the news." + APPENDIX
        )
        self.assertEqual(
            completed["response"]["output"][0]["content"][0]["text"],
            "Here is the news." + APPENDIX,
        )


if __name__ == "__main__":
    unittest.main()
