# Copyright (c) 2026 grok-to-openai-api contributors.
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
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, overload, override
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from websockets.asyncio.client import ClientConnection
from websockets.frames import CloseCode
from websockets.protocol import State

import server
from accounts import AccountPool
from grok_gateway import GatewayError, GrokSession, TurnResult, unrelated_queries

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Iterable

    from websockets.typing import Data, DataLike

    from accounts import Account

_BAD_GATEWAY_STATUS = 502
_DEGRADED_QUARANTINE_SECONDS = 3600
_QUARANTINE_FLOOR_SECONDS = 3599
_TWO_CALLS = 2
_THREE_CALLS = 3
_SIX_CALLS = 6
_SEVEN_CALLS = 7

# --------------------------------------------------------------------- fakes


def write_accounts_file(path: Path, n: int) -> None:
    """Write an accounts file with n cookie accounts.

    Args:
        path: Destination file path.
        n: Number of accounts to write.

    """
    blocks = [
        (
            f"account {i}:\n\n"
            f".grok.com\tTRUE\t/\tTRUE\t1800000000\tsso\tsso-token-{i}\n"
            f".grok.com\tTRUE\t/\tTRUE\t0\tx-userid\tuid-{i:02d}\n"
        )
        for i in range(1, n + 1)
    ]
    path.write_text("\n".join(blocks), encoding="utf-8")


def make_pool(tmp: str, n: int = 3) -> AccountPool:
    """Build a pool backed by a temp accounts file.

    Args:
        tmp: Directory holding the generated accounts file.
        n: Number of accounts to generate.

    Returns:
        A pool loaded synchronously for offline tests.

    """
    p = Path(tmp) / "accounts.txt"
    write_accounts_file(p, n)
    pool = AccountPool(p, cooldown_seconds=300)
    # hot-reload is async; load once synchronously for offline tests
    asyncio.run(pool.reload_if_changed())
    return pool


class FakeWS(ClientConnection):
    """Scripted ws for GrokSession.alive/send/recv without a connection."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        """Stage scripted gateway frames for recv.

        Args:
            frames: Gateway events delivered as text frames in order.

        """
        self.protocol: Any = SimpleNamespace(state=State.OPEN)
        self._frames: list[dict[str, Any]] = list(frames)
        self.sent: list[str] = []

    @overload
    async def recv(self, decode: Literal[True]) -> str: ...

    @overload
    async def recv(self, decode: Literal[False]) -> bytes: ...

    @overload
    async def recv(
        self,
        decode: object = None,
    ) -> Data: ...

    @override
    async def recv(self, decode: object = None) -> Data:
        if not self._frames:
            raise TimeoutError
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
        self,
        code: CloseCode | int = CloseCode.NORMAL_CLOSURE,
        reason: str = "",
    ) -> None:
        return None


def make_scripted_session(frames: list[dict[str, Any]]) -> GrokSession:
    """Build a session replaying scripted gateway frames.

    Args:
        frames: Gateway events delivered in order.

    Returns:
        A session wired to a scripted websocket.

    """
    sess = GrokSession(cookie_header="ck", user_id="uid", model_mode="fast")
    sess.conversation_id = "conv-1"
    sess.ws = FakeWS(frames)
    return sess


def user_item(user_id: str = "umsg") -> dict[str, Any]:
    """Build a user message event.

    Args:
        user_id: Message id for the user item.

    Returns:
        A conversation item event.

    """
    return {
        "event": {
            "type": "conversation.item.added",
            "item": {"role": "user", "id": user_id},
        },
    }


def tool_query(q: str) -> dict[str, Any]:
    """Build a live-shape search query event.

    Live mgw shape: search cards ride response.chunk events.

    Args:
        q: Search query text.

    Returns:
        A response.chunk event carrying the query.

    """
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {"tool_usage_card": {"web_search": {"args": {"query": q}}}},
        },
    }


def tool_query_on_output(q: str) -> dict[str, Any]:
    """Build a legacy-shape search query event.

    Legacy/alternate shape seen on response.grok.output events.

    Args:
        q: Search query text.

    Returns:
        A response.grok.output event carrying the query.

    """
    return {
        "event": {
            "type": "response.grok.output",
            "output": {"tool_usage_card": {"web_search": {"query": q}}},
        },
    }


def text_chunk(t: str) -> dict[str, Any]:
    """Build an assistant text event.

    Args:
        t: Text streamed on the assistant channel.

    Returns:
        A response.chunk event carrying the text.

    """
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {
                "metadata": {"step_id": 0},
                "text": {"text": t, "channel": "CHANNEL_ASSISTANT_RESPONSE"},
            },
        },
    }


def done_event(status: str = "completed") -> dict[str, Any]:
    """Build a turn completion event.

    Args:
        status: Terminal response status.

    Returns:
        A response.done event.

    """
    return {
        "event": {
            "type": "response.done",
            "response": {"id": "resp-1", "status": status},
        },
    }


async def drive(sess: GrokSession, prompt: str) -> GatewayError | list[dict[str, Any]]:
    """Collect ask() events; return the raised GatewayError, if any.

    Args:
        sess: Scripted gateway session to drive.
        prompt: User prompt sent on the turn.

    Returns:
        The raised gateway error, or the collected turn events.

    """
    try:
        return [ev async for ev in sess.ask(prompt, user_text=prompt)]
    except GatewayError as e:
        return e


def gen_chunk(url: str = "users/uid/generated/abc/final.jpg") -> dict[str, Any]:
    """Build a fresh-generation image event.

    Degraded signature: fresh-generation card, no edit of the attachment.

    Args:
        url: Image URL carried by the card.

    Returns:
        A response.chunk event carrying the generation card.

    """
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {
                "render_generated_image": {
                    "image_chunk": {"imageUrl": url, "progress": 100},
                },
            },
        },
    }


def edit_chunk(url: str = "users/uid/generated/abc/final.jpg") -> dict[str, Any]:
    """Build an edited-image event.

    Healthy edit card: the attached image was actually modified.

    Args:
        url: Image URL carried by the card.

    Returns:
        A response.chunk event carrying the edit card.

    """
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {
                "render_edited_image": {
                    "image_chunk": {"imageUrl": url, "progress": 100},
                },
            },
        },
    }


async def drive_attach(
    sess: GrokSession,
    prompt: str,
    attachment_ids: list[str] | None = None,
    collect: list[dict[str, Any]] | None = None,
) -> GatewayError | list[dict[str, Any]]:
    """Collect ask() events for a turn that mentions attachments.

    When `collect` is given, every event is appended there as it arrives,
    including any emitted before a GatewayError aborts the turn.

    Args:
        sess: Scripted gateway session to drive.
        prompt: User prompt sent on the turn.
        attachment_ids: Attachment ids mentioned on the turn.
        collect: Optional sink receiving every event as it arrives.

    Returns:
        The raised gateway error, or the collected turn events.

    """
    out: list[dict[str, Any]] = []
    try:
        async for ev in sess.ask(
            prompt,
            attachment_ids=attachment_ids,
            user_text=prompt,
        ):
            out.append(ev)
            if collect is not None:
                collect.append(ev)
    except GatewayError as e:
        return e
    return out


# -------------------------------------------------------------- detector


class UnrelatedQueriesTests(unittest.TestCase):
    """Pin the unrelated-query detector over tool usage cards."""

    USER = "what is the capital of france"

    def test_two_distinct_unrelated_queries_fail(self) -> None:
        """Verify two distinct unrelated queries fail."""
        if not unrelated_queries(
            ["weather patterns in europe", "best restaurants in tokyo"],
            self.USER,
        ):
            pytest.fail("two distinct unrelated queries should fail")

    def test_single_paraphrased_query_is_tolerated(self) -> None:
        """Verify a single paraphrased query is tolerated."""
        if unrelated_queries(["paris france geography overview"], self.USER):
            pytest.fail("single paraphrased query should be tolerated")

    def test_known_placeholder_alone_fails(self) -> None:
        """Verify a known placeholder query alone fails."""
        if not unrelated_queries(
            ["current information and recent sources"],
            self.USER,
        ):
            pytest.fail("known placeholder query alone should fail")

    def test_related_then_placeholder_fails(self) -> None:
        """Verify a related query plus a placeholder fails."""
        if not unrelated_queries(
            [
                "france capital history",
                "latest updates and authoritative references",
            ],
            self.USER,
        ):
            pytest.fail("related query plus placeholder should fail")

    @staticmethod
    def test_empty_user_text_with_generic_placeholder_fails() -> None:
        """Verify a generic placeholder fails with empty user text."""
        if not unrelated_queries(["current information and recent sources"], ""):
            pytest.fail("generic placeholder should fail with empty user text")

    def test_duplicate_unrelated_queries_fail(self) -> None:
        """Verify duplicate unrelated queries fail."""
        if not unrelated_queries(["weather patterns in europe"] * 3, self.USER):
            pytest.fail("duplicate unrelated queries should fail")

    @staticmethod
    def test_incident_style_prompts_detected() -> None:
        """Verify incident-style short prompts are detected."""
        # Incident regression: short-prompt messages ("give this cat a hat")
        # whose only >=4-char tokens are "give"/"this"; placeholder queries
        # about other topics share none of them.
        if not unrelated_queries(
            ["bonsai tree care basics", "diy treehouse building plans"],
            "give this cat a hat",
        ):
            pytest.fail("incident-style prompts should be detected")
        # A single unrelated query stays tolerated even for short prompts.
        if unrelated_queries(["hat styles for cats"], "give this cat a hat"):
            pytest.fail("single unrelated query should stay tolerated")


# ------------------------------------------------------------- quarantine


class QuarantineTests(unittest.TestCase):
    """Pin the one-hour pool quarantine for degraded accounts."""

    @override
    def setUp(self) -> None:
        """Build a two-account pool in a temporary directory."""
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pool = make_pool(self._tmp.name, n=2)

    def acc(self, i: int) -> Account:
        """Return the i-th account snapshot entry.

        Args:
            i: Index into the pool snapshot.

        Returns:
            The account at position ``i``.

        """
        return self.pool.snapshot()[i]

    def test_degraded_failure_quarantines_for_an_hour(self) -> None:
        """Verify a degraded failure quarantines for an hour."""
        now = time.time()
        self.pool.release_fail(self.acc(0), "degraded")
        a = self.acc(0)
        if a.available():
            pytest.fail("degraded account should be unavailable")
        if a.cooldown_until - now < _QUARANTINE_FLOOR_SECONDS:
            pytest.fail("degraded cooldown should last about an hour")
        if a.degraded_until - now < _QUARANTINE_FLOOR_SECONDS:
            pytest.fail("degraded quarantine should last about an hour")

    def test_release_ok_does_not_lift_quarantine(self) -> None:
        """Verify a successful release does not lift quarantine."""
        self.pool.release_fail(self.acc(0), "degraded")
        self.pool.release_ok(self.acc(0))
        a = self.acc(0)
        if a.available():
            pytest.fail("quarantined account should stay unavailable")
        if a.degraded_until <= time.time():
            pytest.fail("successful release should not lift quarantine")

    def test_acquire_skips_degraded_account(self) -> None:
        """Verify acquire skips a degraded account."""
        self.pool.release_fail(self.acc(0), "degraded")
        acc = self.pool.acquire()
        if acc is None:
            pytest.fail("healthy account should be acquired")
        if acc.key != self.acc(1).key:
            pytest.fail("degraded account should be skipped")

    def test_re_admitted_after_deadline(self) -> None:
        """Verify a quarantined account is re-admitted after the deadline."""
        a = self.acc(0)
        self.pool.release_fail(a, "degraded")
        real_time = time.time
        with patch("accounts.time.time", lambda: real_time() + 3700):
            if not a.available():
                pytest.fail("account should be available past the deadline")
            if self.pool.acquire() is not a:
                pytest.fail("account should be re-admitted past the deadline")

    def test_generic_failure_keeps_default_cooldown(self) -> None:
        """Verify a generic failure keeps the default cooldown."""
        now = time.time()
        self.pool.release_fail(self.acc(0), "generic")
        a = self.acc(0)
        if a.available():
            pytest.fail("failed account should be cooling down")
        if a.cooldown_until - now >= _DEGRADED_QUARANTINE_SECONDS:
            pytest.fail("generic failure should keep the default cooldown")
        # Threshold comparison, not != 0.0: RUF069 bans float equality and
        # degraded_until is 0.0-initialized and only max-assigned forward.
        if a.degraded_until > 0.0:
            pytest.fail("generic failure should not set a degraded quarantine")


# ------------------------------------------------------ ask()-level guard


class AskDetectionTests(unittest.TestCase):
    """Pin the ask()-level degraded-turn guard."""

    PROMPT = "what is the capital of france"

    def test_completes_when_tool_queries_related(self) -> None:
        """Verify related tool queries let the turn complete."""
        frames = [
            user_item(),
            tool_query("paris france geography overview"),
            text_chunk("Paris"),
            done_event(),
        ]
        out = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        if not isinstance(out, list):
            pytest.fail("related turn should complete without error")
        if out[-1]["type"] != "done":
            pytest.fail("related turn should end with done")
        if out[-1]["result"].text != "Paris":
            pytest.fail("related turn should stream the answer text")

    def test_two_unrelated_queries_abort_before_salad_streams(self) -> None:
        """Verify two unrelated queries abort before salad streams."""
        frames = [
            user_item(),
            tool_query("weather patterns in europe"),
            tool_query("best restaurants in tokyo"),
            text_chunk("word"),
            text_chunk("salad"),
            done_event(),
        ]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        if not isinstance(err, GatewayError):
            pytest.fail("two unrelated queries should abort the turn")
        if err.kind != "degraded":
            pytest.fail("abort should be flagged as degraded")

    def test_single_known_placeholder_aborts(self) -> None:
        """Verify a single known placeholder aborts."""
        frames = [
            user_item(),
            tool_query("current information and recent sources"),
            done_event(),
        ]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        if not isinstance(err, GatewayError):
            pytest.fail("known placeholder should abort the turn")
        if err.kind != "degraded":
            pytest.fail("abort should be flagged as degraded")

    def test_duplicate_unrelated_queries_abort(self) -> None:
        """Verify duplicate unrelated queries abort."""
        frames = [
            user_item(),
            tool_query("weather patterns in europe"),
            tool_query("weather patterns in europe"),
            text_chunk("ok"),
            done_event(),
        ]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        if not isinstance(err, GatewayError):
            pytest.fail("duplicate unrelated queries should abort the turn")
        if err.kind != "degraded":
            pytest.fail("abort should be flagged as degraded")

    def test_flat_query_variant_detected(self) -> None:
        """Verify the flat query variant is detected."""
        frames = [
            user_item(),
            {
                "event": {
                    "type": "response.chunk",
                    "chunk": {
                        "tool_usage_card": {
                            "web_search": {"query": "weather patterns in europe"},
                        },
                    },
                },
            },
            {
                "event": {
                    "type": "response.chunk",
                    "chunk": {
                        "tool_usage_card": {
                            "web_search": {"query": "best restaurants in tokyo"},
                        },
                    },
                },
            },
            done_event(),
        ]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        if not isinstance(err, GatewayError):
            pytest.fail("flat query variant should abort the turn")
        if err.kind != "degraded":
            pytest.fail("abort should be flagged as degraded")

    def test_output_event_queries_count_too(self) -> None:
        """Verify output-event queries count too."""
        frames = [
            user_item(),
            tool_query_on_output("weather patterns in europe"),
            tool_query("best restaurants in tokyo"),
            done_event(),
        ]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        if not isinstance(err, GatewayError):
            pytest.fail("output-event queries should abort the turn")
        if err.kind != "degraded":
            pytest.fail("abort should be flagged as degraded")


def header_chunk(t: str) -> dict[str, Any]:
    """Build a notetaker header event.

    Args:
        t: Header text.

    Returns:
        A response.chunk event carrying the header.

    """
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {
                "text": {"text": t, "channel": "CHANNEL_ASSISTANT_NOTETAKER_HEADER"},
            },
        },
    }


def thinking_chunk(t: str) -> dict[str, Any]:
    """Build a thinking-channel event.

    Args:
        t: Thinking text.

    Returns:
        A response.chunk event carrying the thinking text.

    """
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {
                "text": {"text": t, "channel": "CHANNEL_ASSISTANT_THINKING"},
            },
        },
    }


def summary_chunk(t: str) -> dict[str, Any]:
    """Build a notetaker summary event.

    Args:
        t: Summary text.

    Returns:
        A response.chunk event carrying the summary.

    """
    return {
        "event": {
            "type": "response.chunk",
            "chunk": {
                "text": {"text": t, "channel": "CHANNEL_ASSISTANT_NOTETAKER_SUMMARY"},
            },
        },
    }


class ChannelRoutingTests(unittest.TestCase):
    """Pin channel routing for header, thinking, and summary text."""

    PROMPT = "write a two-sentence story about a robot"

    def test_header_chrome_dropped_and_turn_completes(self) -> None:
        """Verify header chrome is dropped and the turn completes."""
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
        if not isinstance(out, list):
            pytest.fail("header-only turn should complete without error")
        if out[-1]["type"] != "done":
            pytest.fail("header-only turn should end with done")
        if out[-1]["result"].text != "ok":
            pytest.fail("header-only turn should stream the answer text")
        if out[-1]["result"].reasoning:
            pytest.fail("header chrome should not count as reasoning")
        kinds = [e["type"] for e in out]
        if "reasoning_delta" in kinds:
            pytest.fail("header chrome should not yield reasoning deltas")
        deltas = [e["text"] for e in out if e["type"] == "text_delta"]
        if deltas != ["ok"]:
            pytest.fail("only the answer text should stream as text deltas")

    def test_thinking_channel_yields_reasoning(self) -> None:
        """Verify the thinking channel yields reasoning."""
        frames = [
            user_item(),
            thinking_chunk("let me think"),
            text_chunk("hi"),
            done_event(),
        ]
        out = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        if not isinstance(out, list):
            pytest.fail("thinking turn should complete without error")
        if out[-1]["result"].text != "hi":
            pytest.fail("thinking turn should stream the answer text")
        if "let me think" not in out[-1]["result"].reasoning:
            pytest.fail("thinking text should land in reasoning")
        deltas = [e["text"] for e in out if e["type"] == "reasoning_delta"]
        if "let me think" not in deltas:
            pytest.fail("thinking text should stream as reasoning deltas")

    def test_summary_channel_yields_reasoning(self) -> None:
        """Verify the summary channel yields reasoning."""
        frames = [
            user_item(),
            summary_chunk("key points considered"),
            text_chunk("hi"),
            done_event(),
        ]
        out = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        if not isinstance(out, list):
            pytest.fail("summary turn should complete without error")
        if out[-1]["result"].text != "hi":
            pytest.fail("summary turn should stream the answer text")
        if "key points considered" not in out[-1]["result"].reasoning:
            pytest.fail("summary text should land in reasoning")


# --------------------------------------------------------------- failover


class FailoverTests(unittest.TestCase):
    """Pin request failover across accounts for degraded turns."""

    @override
    def setUp(self) -> None:
        """Build a two-account pool and clear cached sessions."""
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pool = make_pool(self._tmp.name, n=2)
        server.SESSIONS.clear()

    @override
    def tearDown(self) -> None:
        """Clear cached sessions."""
        server.SESSIONS.clear()

    def test_degraded_turn_fails_over_to_next_account(self) -> None:
        """Verify a degraded turn fails over to the next account."""
        pool = self.pool
        calls: list[str] = []

        async def fake_run(
            sess: GrokSession,
            *_args: object,
            **_kwargs: object,
        ) -> tuple[TurnResult, list[dict[str, Any]]]:
            await asyncio.sleep(0)
            calls.append(sess.cookie_header)
            if "sso-token-1" in sess.cookie_header:
                kind = "degraded"
                msg = "gateway searched unrelated content"
                raise GatewayError(kind, msg)
            return TurnResult(text="ok"), []

        with (
            patch.object(server, "pool", new=pool),
            patch.object(server, "run_session_turn", new=fake_run),
        ):
            acc, result, _events, _state = asyncio.run(
                server.pick_account_and_turn(None, ["hi"], mode="fast", prompt="hi"),
            )

        if result.text != "ok":
            pytest.fail("failover turn should serve the answer")
        if acc.key != "u:uid-02":
            pytest.fail("failover turn should land on the second account")
        # exactly two attempts: first account failed, second served the turn
        if len(calls) != _TWO_CALLS:
            pytest.fail("failover should attempt exactly two accounts")
        degraded = next(a for a in pool.snapshot() if a.key == "u:uid-01")
        if degraded.available():
            pytest.fail("degraded account should be quarantined")
        if degraded.degraded_until <= time.time():
            pytest.fail("degraded account should hold a future quarantine")

    def test_all_degraded_surfaces_502_not_salad(self) -> None:
        """Verify an all-degraded pool surfaces a 502, not salad."""
        pool = self.pool

        async def fake_run(
            *_args: object,
            **_kwargs: object,
        ) -> tuple[TurnResult, list[dict[str, Any]]]:
            await asyncio.sleep(0)
            kind = "degraded"
            msg = "gateway searched unrelated content"
            raise GatewayError(kind, msg)

        with (
            patch.object(server, "pool", new=pool),
            patch.object(server, "run_session_turn", new=fake_run),
            pytest.raises(HTTPException) as ctx,
        ):
            asyncio.run(
                server.pick_account_and_turn(None, ["hi"], mode="fast", prompt="hi"),
            )
        if ctx.value.status_code != _BAD_GATEWAY_STATUS:
            pytest.fail("all-degraded pool should surface a 502")

    def test_failover_tries_every_account_before_raising(self) -> None:
        """Verify failover tries every account before raising."""
        pool = make_pool(self._tmp.name, n=7)
        calls: list[str] = []

        async def fake_run(
            sess: GrokSession,
            *_args: object,
            **_kwargs: object,
        ) -> tuple[TurnResult, list[dict[str, Any]]]:
            await asyncio.sleep(0)
            calls.append(sess.cookie_header)
            kind = "degraded"
            msg = "gateway searched unrelated content"
            raise GatewayError(kind, msg)

        with (
            patch.object(server, "pool", new=pool),
            patch.object(server, "run_session_turn", new=fake_run),
            pytest.raises(HTTPException) as ctx,
        ):
            asyncio.run(
                server.pick_account_and_turn(None, ["hi"], mode="fast", prompt="hi"),
            )

        # every account tried exactly once — even past the old 5-attempt cap
        if len(calls) != _SEVEN_CALLS:
            pytest.fail("failover should try every account exactly once")
        if len(set(calls)) != _SEVEN_CALLS:
            pytest.fail("failover should not retry any account")
        if ctx.value.status_code != _BAD_GATEWAY_STATUS:
            pytest.fail("exhausted pool should surface a 502")

    def test_degraded_quarantined_accounts_get_last_resort_attempt(self) -> None:
        """Verify quarantined accounts get a last-resort attempt."""
        pool = make_pool(self._tmp.name, n=3)
        for a in pool.snapshot():
            pool.release_fail(a, "degraded")
        calls: list[str] = []

        async def fake_run(
            sess: GrokSession,
            *_args: object,
            **_kwargs: object,
        ) -> tuple[TurnResult, list[dict[str, Any]]]:
            await asyncio.sleep(0)
            calls.append(sess.cookie_header)
            if len(calls) < _THREE_CALLS:
                kind = "degraded"
                msg = "gateway searched unrelated content"
                raise GatewayError(kind, msg)
            return TurnResult(text="ok"), []

        with (
            patch.object(server, "pool", new=pool),
            patch.object(server, "run_session_turn", new=fake_run),
        ):
            _acc, result, _events, _state = asyncio.run(
                server.pick_account_and_turn(None, ["hi"], mode="fast", prompt="hi"),
            )

        if result.text != "ok":
            pytest.fail("last-resort attempt should serve the answer")
        if len(calls) != _THREE_CALLS:
            pytest.fail("last resort should attempt three accounts")
        if len(set(calls)) != _THREE_CALLS:
            pytest.fail("last resort should not retry any account")

    def test_stream_failover_tries_every_account_before_raising(self) -> None:
        """Verify stream failover tries every account before raising."""
        pool = make_pool(self._tmp.name, n=6)
        calls: list[str] = []

        async def fake_stream(
            sess: GrokSession,
            *_args: object,
            **_kwargs: object,
        ) -> AsyncIterator[dict[str, Any]]:
            await asyncio.sleep(0)
            calls.append(sess.cookie_header)
            kind = "upstream"
            msg = "boom"
            raise GatewayError(kind, msg)
            if False:
                yield {"type": "text_delta", "text": "unreachable"}

        async def consume() -> None:
            async for _ in server.pick_account_and_stream_turn(
                None,
                ["hi"],
                mode="fast",
                prompt="hi",
            ):
                pass

        with (
            patch.object(server, "pool", new=pool),
            patch.object(server, "stream_session_turn", new=fake_stream),
            pytest.raises(HTTPException) as ctx,
        ):
            asyncio.run(consume())

        if len(calls) != _SIX_CALLS:
            pytest.fail("stream failover should try every account exactly once")
        if len(set(calls)) != _SIX_CALLS:
            pytest.fail("stream failover should not retry any account")
        if ctx.value.status_code != _BAD_GATEWAY_STATUS:
            pytest.fail("exhausted pool should surface a 502")

    def test_stream_continuation_guard_prevents_zero_cooldown_loop(self) -> None:
        """Verify the continuation guard prevents a zero-cooldown loop."""
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
            account_key=acc0.key,
            grok=sess0,
            user_chain=["hi"],
        )
        calls: list[str] = []

        async def fake_stream(
            sess: GrokSession,
            *_args: object,
            **_kwargs: object,
        ) -> AsyncIterator[dict[str, Any]]:
            await asyncio.sleep(0)
            calls.append(sess.cookie_header)
            kind = "upstream"
            msg = "boom"
            raise GatewayError(kind, msg)
            if False:
                yield {"type": "text_delta", "text": "unreachable"}

        async def consume() -> None:
            async for _ in server.pick_account_and_stream_turn(
                "sess-key",
                ["hi"],
                mode="fast",
                prompt="hi",
            ):
                pass

        with (
            patch.object(server, "pool", new=pool),
            patch.object(server, "stream_session_turn", new=fake_stream),
            pytest.raises(HTTPException) as ctx,
        ):
            asyncio.run(asyncio.wait_for(consume(), timeout=10))

        # continuation account + the one untried account, each exactly once
        if len(calls) != _TWO_CALLS:
            pytest.fail("continuation guard should attempt two accounts")
        if len(set(calls)) != _TWO_CALLS:
            pytest.fail("continuation guard should not retry any account")
        if ctx.value.status_code != _BAD_GATEWAY_STATUS:
            pytest.fail("exhausted pool should surface a 502")


class ImageEditDegradedTests(unittest.TestCase):
    """Pin degraded detection for image-edit turns."""

    PROMPT = "give this cat a hat"

    def test_unrelated_generation_with_attachments_aborts(self) -> None:
        """Verify an unrelated generation with attachments aborts."""
        # Captured live 2026-08-31: a degraded gateway ignores the attached
        # image and the prompt, then completes the turn with a placeholder
        # generation carrying zero reasoning and zero text.
        frames = [user_item(), gen_chunk(), done_event()]
        err = asyncio.run(
            drive_attach(
                make_scripted_session(frames),
                self.PROMPT,
                attachment_ids=["fid"],
            ),
        )
        if not isinstance(err, GatewayError):
            pytest.fail("unrelated generation should abort the turn")
        if err.kind != "degraded":
            pytest.fail("abort should be flagged as degraded")

    def test_edited_image_without_reasoning_still_completes(self) -> None:
        """Verify an edited image without reasoning still completes."""
        # 2026-08-28 incident input B: correct edited image with no visible
        # text. The edit card is the health signal; missing text must not
        # fail the turn.
        frames = [user_item(), edit_chunk(), done_event()]
        out = asyncio.run(
            drive_attach(
                make_scripted_session(frames),
                self.PROMPT,
                attachment_ids=["fid"],
            ),
        )
        if not isinstance(out, list):
            pytest.fail("edited image turn should complete without error")
        imgs = [e for e in out if e["type"] == "image_url"]
        if [e["kind"] for e in imgs] != ["edited"]:
            pytest.fail("edited image should keep the edited kind")
        if out[-1]["type"] != "done":
            pytest.fail("edited image turn should end with done")

    def test_mixed_edit_and_generated_cards_complete(self) -> None:
        """Verify mixed edit and generated cards complete."""
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
                make_scripted_session(frames),
                self.PROMPT,
                attachment_ids=["fid"],
            ),
        )
        if not isinstance(out, list):
            pytest.fail("mixed-card turn should complete without error")
        imgs = [e for e in out if e["type"] == "image_url"]
        if [e["kind"] for e in imgs] != ["edited", "generated"]:
            pytest.fail("image kinds should follow yield order")
        if out[-1]["type"] != "done":
            pytest.fail("mixed-card turn should end with done")

    def test_degraded_abort_emits_no_client_visible_images(self) -> None:
        """Verify a degraded abort emits no client-visible images."""
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
            ),
        )
        if not isinstance(err, GatewayError):
            pytest.fail("degraded generation should abort the turn")
        if err.kind != "degraded":
            pytest.fail("abort should be flagged as degraded")
        if [e for e in seen if e["type"] == "image_url"] != []:
            pytest.fail("degraded abort should emit no images")

    @staticmethod
    def test_first_seen_url_kind_wins() -> None:
        """Verify the first-seen URL kind wins."""
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
                make_scripted_session(frames),
                "draw a cat",
                attachment_ids=None,
            ),
        )
        if isinstance(out, GatewayError):
            pytest.fail("re-labeled turn should complete without error")
        imgs = [e for e in out if e["type"] == "image_url"]
        if [e["kind"] for e in imgs] != ["generated"]:
            pytest.fail("first-seen kind should win")

    def test_reasoning_with_generated_image_completes(self) -> None:
        """Verify reasoning with a generated image completes."""
        # The guard requires BOTH zero reasoning and zero text; a turn that
        # reasons about a generation is a model choice, not a degraded turn.
        frames = [
            user_item(),
            {
                "event": {
                    "type": "response.chunk",
                    "chunk": {
                        "text": {"text": "adding a hat...", "channel": "NOTETAKER"},
                    },
                },
            },
            gen_chunk(),
            done_event(),
        ]
        out = asyncio.run(
            drive_attach(
                make_scripted_session(frames),
                self.PROMPT,
                attachment_ids=["fid"],
            ),
        )
        if not isinstance(out, list):
            pytest.fail("reasoned generation should complete without error")
        if out[-1]["type"] != "done":
            pytest.fail("reasoned generation should end with done")

    def test_output_event_generated_image_aborts(self) -> None:
        """Verify an output-event generated image aborts."""
        # Second live image shape: the asset rides response.grok.output
        # (out.generated_image) instead of a response.chunk card.
        frames = [
            user_item(),
            {
                "event": {
                    "type": "response.grok.output",
                    "output": {
                        "generated_image": {
                            "imageUrl": "users/uid/generated/abc/final.jpg",
                        },
                    },
                },
            },
            done_event(),
        ]
        err = asyncio.run(
            drive_attach(
                make_scripted_session(frames),
                self.PROMPT,
                attachment_ids=["fid"],
            ),
        )
        if not isinstance(err, GatewayError):
            pytest.fail("output-event generation should abort the turn")
        if err.kind != "degraded":
            pytest.fail("abort should be flagged as degraded")

    @staticmethod
    def test_fresh_generation_without_attachments_not_flagged() -> None:
        """Verify a fresh generation without attachments is not flagged."""
        # Text-to-image through the gateway legitimately has no edit card.
        frames = [user_item(), gen_chunk(), done_event()]
        out = asyncio.run(
            drive_attach(
                make_scripted_session(frames),
                "draw a cat",
                attachment_ids=None,
            ),
        )
        if not isinstance(out, list):
            pytest.fail("fresh generation should complete without error")
        if out[-1]["type"] != "done":
            pytest.fail("fresh generation should end with done")

    @staticmethod
    def test_text_answer_with_rementioned_attachment_not_flagged() -> None:
        """Verify a text answer with a rementioned attachment is not flagged."""
        # Follow-up turns re-mention remembered attachments for context; a
        # plain text answer must not be treated as a degraded image turn.
        frames = [user_item(), text_chunk("A tabby, I think."), done_event()]
        out = asyncio.run(
            drive_attach(
                make_scripted_session(frames),
                "what breed is this?",
                attachment_ids=["fid"],
            ),
        )
        if not isinstance(out, list):
            pytest.fail("text answer should complete without error")
        if out[-1]["type"] != "done":
            pytest.fail("text answer should end with done")


if __name__ == "__main__":
    unittest.main()
