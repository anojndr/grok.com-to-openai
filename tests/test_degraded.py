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
import time
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import server
from accounts import AccountPool
from grok_gateway import GatewayError, GrokSession, TurnResult
from grok_gateway import unrelated_queries


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


class FakeWS:
    """Scripted ws for GrokSession.alive/send/recv without a connection."""

    def __init__(self, frames: list[dict]):
        from websockets.protocol import State
        self.protocol = SimpleNamespace(state=State.OPEN)
        self._frames = list(frames)
        self.sent: list[str] = []

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        if not self._frames:
            raise asyncio.TimeoutError()
        # real websockets deliver text frames; GrokSession.ask json.loads them
        return json.dumps(self._frames.pop(0))


def make_scripted_session(frames: list[dict]) -> GrokSession:
    sess = GrokSession(cookie_header="ck", user_id="uid", model_mode="fast")
    sess.conversation_id = "conv-1"
    sess.ws = FakeWS(frames)
    return sess


def user_item(user_id: str = "umsg") -> dict:
    return {"event": {"type": "conversation.item.added",
                      "item": {"role": "user", "id": user_id}}}


def tool_query(q: str) -> dict:
    """Live mgw shape: search cards ride response.chunk events."""
    return {"event": {"type": "response.chunk",
                      "chunk": {"tool_usage_card":
                                {"web_search": {"args": {"query": q}}}}}}


def tool_query_on_output(q: str) -> dict:
    """Legacy/alternate shape seen on response.grok.output events."""
    return {"event": {"type": "response.grok.output",
                      "output": {"tool_usage_card":
                                 {"web_search": {"query": q}}}}}


def text_chunk(t: str) -> dict:
    return {"event": {"type": "response.chunk",
                      "chunk": {"metadata": {"step_id": 0},
                                "text": {"text": t,
                                         "channel": "CHANNEL_ASSISTANT_RESPONSE"}}}}


def done_event(status: str = "completed") -> dict:
    return {"event": {"type": "response.done",
                      "response": {"id": "resp-1", "status": status}}}


async def drive(sess: GrokSession, prompt: str):
    """Collect ask() events; return the raised GatewayError, if any."""
    out = []
    try:
        async for ev in sess.ask(prompt, user_text=prompt):
            out.append(ev)
    except GatewayError as e:
        return e
    return out


# -------------------------------------------------------------- detector

class UnrelatedQueriesTests(unittest.TestCase):
    USER = "what is the capital of france"

    def test_two_distinct_unrelated_queries_fail(self):
        self.assertTrue(unrelated_queries(
            ["weather patterns in europe", "best restaurants in tokyo"], self.USER))

    def test_single_paraphrased_query_is_tolerated(self):
        self.assertFalse(unrelated_queries(
            ["paris france geography overview"], self.USER))

    def test_known_placeholder_alone_fails(self):
        self.assertTrue(unrelated_queries(
            ["current information and recent sources"], self.USER))

    def test_related_then_placeholder_fails(self):
        self.assertTrue(unrelated_queries(
            ["france capital history", "latest updates and authoritative references"],
            self.USER))

    def test_empty_user_text_with_generic_placeholder_fails(self):
        self.assertTrue(unrelated_queries(
            ["current information and recent sources"], ""))

    def test_duplicate_unrelated_queries_fail(self):
        self.assertTrue(unrelated_queries(
            ["weather patterns in europe"] * 3, self.USER))

    def test_incident_style_prompts_detected(self):
        # Incident regression: short-prompt messages ("give this cat a hat")
        # whose only >=4-char tokens are "give"/"this"; placeholder queries
        # about other topics share none of them.
        self.assertTrue(unrelated_queries(
            ["bonsai tree care basics", "diy treehouse building plans"],
            "give this cat a hat"))
        # A single unrelated query stays tolerated even for short prompts.
        self.assertFalse(unrelated_queries(
            ["hat styles for cats"], "give this cat a hat"))



# ------------------------------------------------------------- quarantine

class QuarantineTests(unittest.TestCase):
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
        frames = [user_item(),
                  tool_query("paris france geography overview"),
                  text_chunk("Paris"),
                  done_event()]
        out = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(out, list)
        self.assertEqual(out[-1]["type"], "done")
        self.assertEqual(out[-1]["result"].text, "Paris")

    def test_two_unrelated_queries_abort_before_salad_streams(self):
        frames = [user_item(),
                  tool_query("weather patterns in europe"),
                  tool_query("best restaurants in tokyo"),
                  text_chunk("word"),
                  text_chunk("salad"),
                  done_event()]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(err, GatewayError)
        self.assertEqual(err.kind, "degraded")

    def test_single_known_placeholder_aborts(self):
        frames = [user_item(),
                  tool_query("current information and recent sources"),
                  done_event()]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(err, GatewayError)
        self.assertEqual(err.kind, "degraded")

    def test_duplicate_unrelated_queries_abort(self):
        frames = [user_item(),
                  tool_query("weather patterns in europe"),
                  tool_query("weather patterns in europe"),
                  text_chunk("ok"),
                  done_event()]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(err, GatewayError)
        self.assertEqual(err.kind, "degraded")
    def test_flat_query_variant_detected(self):
        frames = [user_item(),
                  {"event": {"type": "response.chunk",
                             "chunk": {"tool_usage_card":
                                       {"web_search": {"query": "weather patterns in europe"}}}}},
                  {"event": {"type": "response.chunk",
                             "chunk": {"tool_usage_card":
                                       {"web_search": {"query": "best restaurants in tokyo"}}}}},
                  done_event()]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(err, GatewayError)
        self.assertEqual(err.kind, "degraded")

    def test_output_event_queries_count_too(self):
        frames = [user_item(),
                  tool_query_on_output("weather patterns in europe"),
                  tool_query("best restaurants in tokyo"),
                  done_event()]
        err = asyncio.run(drive(make_scripted_session(frames), self.PROMPT))
        self.assertIsInstance(err, GatewayError)
        self.assertEqual(err.kind, "degraded")


# --------------------------------------------------------------- failover

class FailoverTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pool = make_pool(self._tmp.name, n=2)
        server.SESSIONS.clear()

    def tearDown(self):
        server.SESSIONS.clear()

    def test_degraded_turn_fails_over_to_next_account(self):
        pool = self.pool
        calls: list[str] = []

        async def fake_run(sess, prompt, **kwargs):
            calls.append(sess.cookie_header)
            if "sso-token-1" in sess.cookie_header:
                raise GatewayError("degraded", "gateway searched unrelated content")
            return TurnResult(text="ok"), []

        real_pool = server.pool
        real_run = server.run_session_turn
        server.pool = pool
        server.run_session_turn = fake_run
        try:
            acc, result, events, state = asyncio.run(
                server.pick_account_and_turn(None, ["hi"], mode="fast",
                                             prompt="hi"))
        finally:
            server.pool = real_pool
            server.run_session_turn = real_run

        self.assertEqual(result.text, "ok")
        self.assertEqual(acc.key, "u:uid-02")
        # exactly two attempts: first account failed, second served the turn
        self.assertEqual(len(calls), 2)
        degraded = next(a for a in pool.snapshot() if a.key == "u:uid-01")
        self.assertFalse(degraded.available())
        self.assertGreater(degraded.degraded_until, time.time())

    def test_all_degraded_surfaces_502_not_salad(self):
        pool = self.pool

        async def fake_run(sess, prompt, **kwargs):
            raise GatewayError("degraded", "gateway searched unrelated content")

        real_pool = server.pool
        real_run = server.run_session_turn
        server.pool = pool
        server.run_session_turn = fake_run
        try:
            with self.assertRaises(Exception) as ctx:
                asyncio.run(server.pick_account_and_turn(
                    None, ["hi"], mode="fast", prompt="hi"))
        finally:
            server.pool = real_pool
            server.run_session_turn = real_run
        self.assertEqual(ctx.exception.status_code, 502)


if __name__ == "__main__":
    unittest.main()
