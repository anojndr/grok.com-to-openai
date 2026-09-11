"""Degraded-turn detection, account quarantine, and account failover.

Some grok gateways intermittently run their web-search tool against
placeholder queries unrelated to the user's message and then summarize
that unrelated content into word salad that still parses as HTTP 200.
These tests pin:

  * the early detector over response.grok.output tool_usage_card events,
  * the one-hour pool quarantine for accounts caught serving degraded
    turns (and that successful releases do not lift it),
  * request failover: a turn aborted as degraded is retried on another
    account and the offending account stops receiving traffic.

All tests are offline; gateway frames are scripted.
Run: python3 -m unittest -v tests.test_degraded
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Literal, overload, override
from unittest.mock import patch

from websockets.asyncio.client import ClientConnection
from websockets.frames import CloseCode
from websockets.typing import Data, DataLike

import server
from accounts import AccountPool
from grok_gateway import GatewayError, GrokSession, TurnResult, unrelated_queries

# --------------------------------------------------------------------- fakes


def write_accounts_file(path: Path, n: int) -> None:
    blocks = []
    for i in range(1, n + 1):
        blocks.append(
            f"account {i}:\n\n"
            f".grok.com\tTRUE\t/\tTRUE\t1800000000\tsso\tsso-token-{i}\n"
            f".grok.com\tTRUE\t/\tTRUE\t0\tx-userid\tuid-{i:02d}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def make_pool(tmp: str, n: int = 3) -> AccountPool:
    p = Path(tmp) / "accounts.txt"
    write_accounts_file(p, n)
    pool = AccountPool(p, cooldown_seconds=300)
    # hot-reload is async; load once synchronously for offline tests
    asyncio.run(pool.reload_if_changed())
    return pool


class FakeWS(ClientConnection):
    """Scripted ws for GrokSession.alive/send/recv without a connection."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        from websockets.protocol import State

        self.protocol: Any = SimpleNamespace(state=State.OPEN)
        self._frames: list[dict[str, Any]] = list(frames)
        self.sent: list[str] = []

    @overload
    async def recv(self, decode: Literal[True]) -> str: ...

    @overload
    async def recv(self, decode: Literal[False]) -> bytes: ...

    @overload
    async def recv(self, decode: bool | None = None) -> Data: ...

    @override
    async def recv(self, decode: bool | None = None) -> Data:
        if not self._frames:
            raise TimeoutError()
        # real websockets deliver text frames; GrokSession.ask json.loads them
        return json.dumps(self._frames.pop(0))

    @override
    async def send(
        self,
        message: DataLike | Iterable[DataLike] | AsyncIterable[DataLike],
        *,
        text: bool | None = None,
    ) -> None:
        if isinstance(message, str):
            self.sent.append(message)
        else:
            self.sent.append(str(message))

    @override
    async def close(
        self, code: CloseCode | int = CloseCode.NORMAL_CLOSURE, reason: str = ""
    ) -> None:
        return None


def make_scripted_session(frames: list[dict[str, Any]]) -> GrokSession:
    sess = GrokSession(cookie_header="ck", user_id="uid", model_mode="fast")
    sess.conversation_id = "conv-1"
    sess.ws = FakeWS(frames)
    return sess


def user_item(user_id: str = "umsg") -> dict[str, Any]:
    return {
        "event": {
            "type": "conversation.item.added",
            "item": {"role": "user", "id": user_id},
        }
    }


def tool_query(q: str) -> dict[str, Any]:
    """Live mgw shape: search cards ride response.chunk events."""
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {"tool_usage_card": {"web_search": {"args": {"query": q}}}},
        }
    }


def tool_query_on_output(q: str) -> dict[str, Any]:
    """Legacy/alternate shape seen on response.grok.output events."""
    return {
        "event": {
            "type": "response.grok.output",
            "output": {"tool_usage_card": {"web_search": {"query": q}}},
        }
    }


def text_chunk(t: str) -> dict[str, Any]:
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {
                "metadata": {"step_id": 0},
                "text": {"text": t, "channel": "CHANNEL_ASSISTANT_RESPONSE"},
            },
        }
    }


def done_event(status: str = "completed") -> dict[str, Any]:
    return {
        "event": {
            "type": "response.done",
            "response": {"id": "resp-1", "status": status},
        }
    }


async def drive(sess: GrokSession, prompt: str):
    """Collect ask() events; return the raised GatewayError, if any."""
    out = []
    try:
        async for ev in sess.ask(prompt, user_text=prompt):
            out.append(ev)
    except GatewayError as e:
        return e
    return out


def gen_chunk(url: str = "users/uid/generated/abc/final.jpg") -> dict[str, Any]:
    """Degraded signature: fresh-generation card, no edit of the attachment."""
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {
                "render_generated_image": {
                    "image_chunk": {"imageUrl": url, "progress": 100}
                }
            },
        }
    }


def edit_chunk(url: str = "users/uid/generated/abc/final.jpg") -> dict[str, Any]:
    """Healthy edit card: the attached image was actually modified."""
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {
                "render_edited_image": {
                    "image_chunk": {"imageUrl": url, "progress": 100}
                }
            },
        }
    }


async def drive_attach(
    sess: GrokSession,
    prompt: str,
    attachment_ids: list[str] | None = None,
    collect: list[Any] | None = None,
):
    """drive() variant that mentions attachments (image-edit turns).

    When `collect` is given, every event is appended there as it arrives,
    including any emitted before a GatewayError aborts the turn.
    """
    out = []
    try:
        async for ev in sess.ask(
            prompt, attachment_ids=attachment_ids, user_text=prompt
        ):
            out.append(ev)
            if collect is not None:
                collect.append(ev)
    except GatewayError as e:
        return e
    return out


# -------------------------------------------------------------- detector


class UnrelatedQueriesTests(unittest.TestCase):
    USER = "what is the capital of france"

    def test_two_distinct_unrelated_queries_fail(self):
        self.assertTrue(
            unrelated_queries(
                ["weather patterns in europe", "best restaurants in tokyo"], self.USER
            )
        )

    def test_single_paraphrased_query_is_tolerated(self):
        self.assertFalse(
            unrelated_queries(["paris france geography overview"], self.USER)
        )

    def test_known_placeholder_alone_fails(self):
        self.assertTrue(
            unrelated_queries(["current information and recent sources"], self.USER)
        )

    def test_related_then_placeholder_fails(self):
        self.assertTrue(
            unrelated_queries(
                [
                    "france capital history",
                    "latest updates and authoritative references",
                ],
                self.USER,
            )
        )

    def test_empty_user_text_with_generic_placeholder_fails(self):
        self.assertTrue(
            unrelated_queries(["current information and recent sources"], "")
        )

    def test_duplicate_unrelated_queries_fail(self):
        self.assertTrue(
            unrelated_queries(["weather patterns in europe"] * 3, self.USER)
        )

    def test_incident_style_prompts_detected(self):
        # Incident regression: short-prompt messages ("give this cat a hat")
        # whose only >=4-char tokens are "give"/"this"; placeholder queries
        # about other topics share none of them.
        self.assertTrue(
            unrelated_queries(
                ["bonsai tree care basics", "diy treehouse building plans"],
                "give this cat a hat",
            )
        )
        # A single unrelated query stays tolerated even for short prompts.
        self.assertFalse(
            unrelated_queries(["hat styles for cats"], "give this cat a hat")
        )


# ------------------------------------------------------------- quarantine


class QuarantineTests(unittest.TestCase):
    @override
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pool = make_pool(self._tmp.name, n=2)

    def acc(self, i: int):
        return self.pool.snapshot()[i]

    def test_degraded_failure_quarantines_for_an_hour(self):
        now = time.time()
        self.pool.release_fail(self.acc(0), "degraded")
        a = self.acc(0)
        self.assertFalse(a.available())
        self.assertGreaterEqual(a.cooldown_until - now, 3599)
        self.assertGreaterEqual(a.degraded_until - now, 3599)

    def test_release_ok_does_not_lift_quarantine(self):
        self.pool.release_fail(self.acc(0), "degraded")
        self.pool.release_ok(self.acc(0))
        a = self.acc(0)
        self.assertFalse(a.available())
        self.assertGreater(a.degraded_until, time.time())

    def test_acquire_skips_degraded_account(self):
        self.pool.release_fail(self.acc(0), "degraded")
        acc = self.pool.acquire()
        assert acc is not None
        self.assertEqual(acc.key, self.acc(1).key)

    def test_re_admitted_after_deadline(self):
        a = self.acc(0)
        self.pool.release_fail(a, "degraded")
        real_time = time.time
        with patch("accounts.time.time", lambda: real_time() + 3700):
            self.assertTrue(a.available())
            self.assertIs(self.pool.acquire(), a)

    def test_generic_failure_keeps_default_cooldown(self):
        now = time.time()
        self.pool.release_fail(self.acc(0), "generic")
        a = self.acc(0)
        self.assertFalse(a.available())
        self.assertLess(a.cooldown_until - now, 3600)
        self.assertEqual(a.degraded_until, 0.0)


# ------------------------------------------------------ ask()-level guard


class AskDetectionTests(unittest.TestCase):
    PROMPT = "what is the capital of france"

    def test_completes_when_tool_queries_related(self):
        frames = [
            user_item(),
            tool_query("paris france geography overview"),
            text_chunk("Paris"),
            done_event(),
        ]
        out = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(out, list)
        self.assertEqual(out[-1]["type"], "done")
        self.assertEqual(out[-1]["result"].text, "Paris")

    def test_two_unrelated_queries_abort_before_salad_streams(self):
        frames = [
            user_item(),
            tool_query("weather patterns in europe"),
            tool_query("best restaurants in tokyo"),
            text_chunk("word"),
            text_chunk("salad"),
            done_event(),
        ]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(err, GatewayError)
        self.assertEqual(err.kind, "degraded")

    def test_single_known_placeholder_aborts(self):
        frames = [
            user_item(),
            tool_query("current information and recent sources"),
            done_event(),
        ]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(err, GatewayError)
        self.assertEqual(err.kind, "degraded")

    def test_duplicate_unrelated_queries_abort(self):
        frames = [
            user_item(),
            tool_query("weather patterns in europe"),
            tool_query("weather patterns in europe"),
            text_chunk("ok"),
            done_event(),
        ]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(err, GatewayError)
        self.assertEqual(err.kind, "degraded")

    def test_flat_query_variant_detected(self):
        frames = [
            user_item(),
            {
                "event": {
                    "type": "response.chunk",
                    "chunk": {
                        "tool_usage_card": {
                            "web_search": {"query": "weather patterns in europe"}
                        }
                    },
                }
            },
            {
                "event": {
                    "type": "response.chunk",
                    "chunk": {
                        "tool_usage_card": {
                            "web_search": {"query": "best restaurants in tokyo"}
                        }
                    },
                }
            },
            done_event(),
        ]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(err, GatewayError)
        self.assertEqual(err.kind, "degraded")

    def test_output_event_queries_count_too(self):
        frames = [
            user_item(),
            tool_query_on_output("weather patterns in europe"),
            tool_query("best restaurants in tokyo"),
            done_event(),
        ]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(err, GatewayError)
        self.assertEqual(err.kind, "degraded")


def header_chunk(t: str) -> dict[str, Any]:
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {
                "text": {"text": t, "channel": "CHANNEL_ASSISTANT_NOTETAKER_HEADER"},
            },
        }
    }


def thinking_chunk(t: str) -> dict[str, Any]:
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {
                "text": {"text": t, "channel": "CHANNEL_ASSISTANT_THINKING"},
            },
        }
    }


def summary_chunk(t: str) -> dict[str, Any]:
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {
                "text": {"text": t, "channel": "CHANNEL_ASSISTANT_NOTETAKER_SUMMARY"},
            },
        }
    }


class ChannelRoutingTests(unittest.TestCase):
    PROMPT = "write a two-sentence story about a robot"

    def test_header_chrome_dropped_and_turn_completes(self):
        # Live healthy fast turns open with a NOTETAKER_HEADER title
        # ("Thinking about your request"); it is UI chrome, not reasoning,
        # and must not abort the turn as degraded.
        frames = [
            user_item(),
            header_chunk("Thinking about your request"),
            header_chunk("Writing a two-sentence robot story"),
            text_chunk("ok"),
            done_event(),
        ]
        out = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(out, list)
        self.assertEqual(out[-1]["type"], "done")
        self.assertEqual(out[-1]["result"].text, "ok")
        self.assertEqual(out[-1]["result"].reasoning, "")
        kinds = [e["type"] for e in out]
        self.assertNotIn("reasoning_delta", kinds)
        self.assertEqual([e["text"] for e in out if e["type"] == "text_delta"], ["ok"])

    def test_thinking_channel_yields_reasoning(self):
        frames = [
            user_item(),
            thinking_chunk("let me think"),
            text_chunk("hi"),
            done_event(),
        ]
        out = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(out, list)
        self.assertEqual(out[-1]["result"].text, "hi")
        self.assertIn("let me think", out[-1]["result"].reasoning)
        self.assertIn(
            "let me think",
            [e["text"] for e in out if e["type"] == "reasoning_delta"],
        )

    def test_summary_channel_yields_reasoning(self):
        frames = [
            user_item(),
            summary_chunk("key points considered"),
            text_chunk("hi"),
            done_event(),
        ]
        out = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(out, list)
        self.assertEqual(out[-1]["result"].text, "hi")
        self.assertIn("key points considered", out[-1]["result"].reasoning)


# --------------------------------------------------------------- failover


class FailoverTests(unittest.TestCase):
    @override
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pool = make_pool(self._tmp.name, n=2)
        server.SESSIONS.clear()

    @override
    def tearDown(self):
        server.SESSIONS.clear()

    def test_degraded_turn_fails_over_to_next_account(self):
        pool = self.pool
        calls: list[str] = []

        async def fake_run(
            sess: GrokSession,
            prompt: str,
            *,
            attachment_ids: list[str] | None = None,
            system_prompt: str | None = None,
            file_jobs: list[dict[str, Any]] | None = None,
            user_text: str | None = None,
        ) -> tuple[TurnResult, list[dict[str, Any]]]:
            calls.append(sess.cookie_header)
            if "sso-token-1" in sess.cookie_header:
                raise GatewayError("degraded", "gateway searched unrelated content")
            return TurnResult(text="ok"), []

        with (
            patch.object(server, "pool", new=pool),
            patch.object(server, "run_session_turn", new=fake_run),
        ):
            acc, result, _events, _state = asyncio.run(
                server.pick_account_and_turn(None, ["hi"], mode="fast", prompt="hi")
            )

        self.assertEqual(result.text, "ok")
        self.assertEqual(acc.key, "u:uid-02")
        # exactly two attempts: first account failed, second served the turn
        self.assertEqual(len(calls), 2)
        degraded = next(a for a in pool.snapshot() if a.key == "u:uid-01")
        self.assertFalse(degraded.available())
        self.assertGreater(degraded.degraded_until, time.time())

    def test_all_degraded_surfaces_502_not_salad(self):
        pool = self.pool

        async def fake_run(
            sess: GrokSession,
            prompt: str,
            *,
            attachment_ids: list[str] | None = None,
            system_prompt: str | None = None,
            file_jobs: list[dict[str, Any]] | None = None,
            user_text: str | None = None,
        ) -> tuple[TurnResult, list[dict[str, Any]]]:
            raise GatewayError("degraded", "gateway searched unrelated content")

        with (
            patch.object(server, "pool", new=pool),
            patch.object(server, "run_session_turn", new=fake_run),
            self.assertRaises(Exception) as ctx,
        ):
            asyncio.run(
                server.pick_account_and_turn(None, ["hi"], mode="fast", prompt="hi")
            )
        code = getattr(ctx.exception, "status_code", None)
        assert isinstance(code, int)
        self.assertEqual(code, 502)

    def test_failover_tries_every_account_before_raising(self):
        pool = make_pool(self._tmp.name, n=7)
        calls: list[str] = []

        async def fake_run(
            sess: GrokSession,
            prompt: str,
            *,
            attachment_ids: list[str] | None = None,
            system_prompt: str | None = None,
            file_jobs: list[dict[str, Any]] | None = None,
            user_text: str | None = None,
        ) -> tuple[TurnResult, list[dict[str, Any]]]:
            calls.append(sess.cookie_header)
            raise GatewayError("degraded", "gateway searched unrelated content")

        with (
            patch.object(server, "pool", new=pool),
            patch.object(server, "run_session_turn", new=fake_run),
            self.assertRaises(Exception) as ctx,
        ):
            asyncio.run(
                server.pick_account_and_turn(None, ["hi"], mode="fast", prompt="hi")
            )

        # every account tried exactly once — even past the old 5-attempt cap
        self.assertEqual(len(calls), 7)
        self.assertEqual(len(set(calls)), 7)
        code = getattr(ctx.exception, "status_code", None)
        assert isinstance(code, int)
        self.assertEqual(code, 502)

    def test_degraded_quarantined_accounts_get_last_resort_attempt(self):
        pool = make_pool(self._tmp.name, n=3)
        for a in pool.snapshot():
            pool.release_fail(a, "degraded")
        calls: list[str] = []

        async def fake_run(
            sess: GrokSession,
            prompt: str,
            *,
            attachment_ids: list[str] | None = None,
            system_prompt: str | None = None,
            file_jobs: list[dict[str, Any]] | None = None,
            user_text: str | None = None,
        ) -> tuple[TurnResult, list[dict[str, Any]]]:
            calls.append(sess.cookie_header)
            if len(calls) < 3:
                raise GatewayError("degraded", "gateway searched unrelated content")
            return TurnResult(text="ok"), []

        with (
            patch.object(server, "pool", new=pool),
            patch.object(server, "run_session_turn", new=fake_run),
        ):
            _acc, result, _events, _state = asyncio.run(
                server.pick_account_and_turn(None, ["hi"], mode="fast", prompt="hi")
            )

        self.assertEqual(result.text, "ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(set(calls)), 3)

    def test_stream_failover_tries_every_account_before_raising(self):
        pool = make_pool(self._tmp.name, n=6)
        calls: list[str] = []

        async def fake_stream(
            sess: GrokSession,
            prompt: str,
            *,
            attachment_ids: list[str] | None = None,
            system_prompt: str | None = None,
            file_jobs: list[dict[str, Any]] | None = None,
            user_text: str | None = None,
        ) -> AsyncIterator[dict[str, Any]]:
            calls.append(sess.cookie_header)
            raise GatewayError("upstream", "boom")
            if False:
                yield {"type": "text_delta", "text": "unreachable"}

        with (
            patch.object(server, "pool", new=pool),
            patch.object(server, "stream_session_turn", new=fake_stream),
            self.assertRaises(Exception) as ctx,
        ):

            async def consume():
                async for _ in server.pick_account_and_stream_turn(
                    None, ["hi"], mode="fast", prompt="hi"
                ):
                    pass

            asyncio.run(consume())

        self.assertEqual(len(calls), 6)
        self.assertEqual(len(set(calls)), 6)
        code = getattr(ctx.exception, "status_code", None)
        assert isinstance(code, int)
        self.assertEqual(code, 502)

    def test_stream_continuation_guard_prevents_zero_cooldown_loop(self):
        # G2O_COOLDOWN=0: release_fail leaves the account immediately
        # available, so the continuation branch must not re-attempt an
        # already-tried account or the walk never reaches the pool tail.
        pool = make_pool(self._tmp.name, n=2)
        for a in pool.snapshot():
            a.cooldown_until = 0.0
            a.degraded_until = 0.0
        acc0 = pool.snapshot()[0]
        sess0 = GrokSession("sso-token-1", "uid-01")  # no live socket
        sess0.conversation_id = "conv-1"  # makes the continuation branch eligible
        server.SESSIONS["sess-key"] = server.SessionState(
            account_key=acc0.key, grok=sess0, user_chain=["hi"]
        )
        calls: list[str] = []

        async def fake_stream(
            sess: GrokSession,
            prompt: str,
            *,
            attachment_ids: list[str] | None = None,
            system_prompt: str | None = None,
            file_jobs: list[dict[str, Any]] | None = None,
            user_text: str | None = None,
        ) -> AsyncIterator[dict[str, Any]]:
            calls.append(sess.cookie_header)
            raise GatewayError("upstream", "boom")
            if False:
                yield {"type": "text_delta", "text": "unreachable"}

        with (
            patch.object(server, "pool", new=pool),
            patch.object(server, "stream_session_turn", new=fake_stream),
            self.assertRaises(Exception) as ctx,
        ):

            async def consume():
                async for _ in server.pick_account_and_stream_turn(
                    "sess-key", ["hi"], mode="fast", prompt="hi"
                ):
                    pass

            asyncio.run(asyncio.wait_for(consume(), timeout=10))

        # continuation account + the one untried account, each exactly once
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(set(calls)), 2)
        code = getattr(ctx.exception, "status_code", None)
        assert isinstance(code, int)
        self.assertEqual(code, 502)


class ImageEditDegradedTests(unittest.TestCase):
    PROMPT = "give this cat a hat"

    def test_unrelated_generation_with_attachments_aborts(self):
        # Captured live 2026-08-31: a degraded gateway ignores the attached
        # image and the prompt, then completes the turn with a placeholder
        # generation carrying zero reasoning and zero text.
        frames = [user_item(), gen_chunk(), done_event()]
        err = asyncio.run(
            drive_attach(
                make_scripted_session(frames), self.PROMPT, attachment_ids=["fid"]
            )
        )
        self.assertIsInstance(err, GatewayError)
        self.assertEqual(err.kind, "degraded")

    def test_edited_image_without_reasoning_still_completes(self):
        # 2026-08-28 incident input B: correct edited image with no visible
        # text. The edit card is the health signal; missing text must not
        # fail the turn.
        frames = [user_item(), edit_chunk(), done_event()]
        out = asyncio.run(
            drive_attach(
                make_scripted_session(frames), self.PROMPT, attachment_ids=["fid"]
            )
        )
        self.assertIsInstance(out, list)
        imgs = [e for e in out if e["type"] == "image_url"]
        self.assertEqual([e["kind"] for e in imgs], ["edited"])
        self.assertEqual(out[-1]["type"], "done")

    def test_mixed_edit_and_generated_cards_complete(self):
        # A turn streaming both an edit card and a generation card is still a
        # real edit: one "edited" kind exempts every image in the turn, and
        # kinds stay paired with images in yield order.
        frames = [
            user_item(),
            edit_chunk("users/uid/generated/abc/edit.jpg"),
            gen_chunk("users/uid/generated/abc/gen.jpg"),
            done_event(),
        ]
        out = asyncio.run(
            drive_attach(
                make_scripted_session(frames), self.PROMPT, attachment_ids=["fid"]
            )
        )
        self.assertIsInstance(out, list)
        imgs = [e for e in out if e["type"] == "image_url"]
        self.assertEqual([e["kind"] for e in imgs], ["edited", "generated"])
        self.assertEqual(out[-1]["type"], "done")

    def test_degraded_abort_emits_no_client_visible_images(self):
        # Buffering contract: image_url events are held until after the guard,
        # so a degraded abort reaches the client with nothing rendered and the
        # streaming failover replays cleanly on a healthy account.
        frames = [user_item(), gen_chunk(), done_event()]
        seen: list[dict[str, Any]] = []
        err = asyncio.run(
            drive_attach(
                make_scripted_session(frames),
                self.PROMPT,
                attachment_ids=["fid"],
                collect=seen,
            )
        )
        self.assertIsInstance(err, GatewayError)
        self.assertEqual(err.kind, "degraded")
        self.assertEqual([e for e in seen if e["type"] == "image_url"], [])

    def test_first_seen_url_kind_wins(self):
        # First-seen-wins labeling (documented on TurnResult.image_kinds): a
        # URL first surfaced as a generation is never re-labeled "edited" by a
        # later edit card for the same asset.
        frames = [
            user_item(),
            gen_chunk("users/uid/generated/abc/dup.jpg"),
            edit_chunk("users/uid/generated/abc/dup.jpg"),
            done_event(),
        ]
        out = asyncio.run(
            drive_attach(
                make_scripted_session(frames), "draw a cat", attachment_ids=None
            )
        )
        imgs = [e for e in out if e["type"] == "image_url"]
        self.assertEqual([e["kind"] for e in imgs], ["generated"])

    def test_reasoning_with_generated_image_completes(self):
        # The guard requires BOTH zero reasoning and zero text; a turn that
        # reasons about a generation is a model choice, not a degraded turn.
        frames = [
            user_item(),
            {
                "event": {
                    "type": "response.chunk",
                    "chunk": {
                        "text": {"text": "adding a hat...", "channel": "NOTETAKER"}
                    },
                }
            },
            gen_chunk(),
            done_event(),
        ]
        out = asyncio.run(
            drive_attach(
                make_scripted_session(frames), self.PROMPT, attachment_ids=["fid"]
            )
        )
        self.assertIsInstance(out, list)
        self.assertEqual(out[-1]["type"], "done")

    def test_output_event_generated_image_aborts(self):
        # Second live image shape: the asset rides response.grok.output
        # (out.generated_image) instead of a response.chunk card.
        frames = [
            user_item(),
            {
                "event": {
                    "type": "response.grok.output",
                    "output": {
                        "generated_image": {
                            "imageUrl": "users/uid/generated/abc/final.jpg"
                        }
                    },
                }
            },
            done_event(),
        ]
        err = asyncio.run(
            drive_attach(
                make_scripted_session(frames), self.PROMPT, attachment_ids=["fid"]
            )
        )
        self.assertIsInstance(err, GatewayError)
        self.assertEqual(err.kind, "degraded")

    def test_fresh_generation_without_attachments_not_flagged(self):
        # Text-to-image through the gateway legitimately has no edit card.
        frames = [user_item(), gen_chunk(), done_event()]
        out = asyncio.run(
            drive_attach(
                make_scripted_session(frames), "draw a cat", attachment_ids=None
            )
        )
        self.assertIsInstance(out, list)
        self.assertEqual(out[-1]["type"], "done")

    def test_text_answer_with_rementioned_attachment_not_flagged(self):
        # Follow-up turns re-mention remembered attachments for context; a
        # plain text answer must not be treated as a degraded image turn.
        frames = [user_item(), text_chunk("A tabby, I think."), done_event()]
        out = asyncio.run(
            drive_attach(
                make_scripted_session(frames),
                "what breed is this?",
                attachment_ids=["fid"],
            )
        )
        self.assertIsInstance(out, list)
        self.assertEqual(out[-1]["type"], "done")


if __name__ == "__main__":
    unittest.main()
