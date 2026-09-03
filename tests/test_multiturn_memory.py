"""Regression tests: multi-turn conversation memory.

Incident: "remember the number 974" followed by "what number did i ask you
to remember again?" answered "3" — total context loss. Two layers failed:

1. The gateway was told to forget: default_x_grok() sent keep_context=False,
   so continued turns (newest message only + conversation attach) arrived
   with no history server-side.
2. Turns that missed every checkpoint (account failover, restart, prefix
   mismatch) shipped the bare latest message, discarding the transcript the
   OpenAI-stateless client supplied.

Fix: keep_context=True (temporary + memory-disabled stay set so pooled
accounts never persist chats or leak memory across users), and fresh
sessions resend the role-labeled transcript while continuations keep
sending only the newest message (user_text stays the newest message for
degraded-query detection).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from collections.abc import AsyncIterator
from typing import Any, override
from unittest.mock import AsyncMock, patch

from fastapi import Request

import server
from accounts import Account, AccountPool
from grok_gateway import GrokSession, TurnResult, default_x_grok
from session_store import SqliteStore


class FakeRequest(Request):
    """Real Request carrying a canned JSON body (headers via scope)."""

    def __init__(
        self, body: dict[str, Any], headers: dict[str, str] | None = None
    ) -> None:
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [
                (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
            ],
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


class GatewayFlagsTests(unittest.TestCase):
    def test_conversation_context_kept_without_persistent_memory(self):
        xgrok = default_x_grok()
        self.assertTrue(xgrok["keep_context"])
        # Pooled accounts serve unrelated users: chats must stay temporary
        # with long-term memory off, otherwise one user's facts leak sideways.
        self.assertTrue(xgrok["is_temporary"])
        self.assertTrue(xgrok["disable_memory"])


class HistoryPromptTests(unittest.TestCase):
    def test_single_turn_returns_latest_byte_identical(self):
        flat = [{"role": "user", "content": "remember the number 974"}]
        self.assertEqual(
            server.build_history_prompt(flat, "remember the number 974"),
            "remember the number 974",
        )

    def test_followup_transcript_keeps_both_user_messages(self):
        flat = [
            {"role": "user", "content": "remember the number 974"},
            {"role": "assistant", "content": "Got it, 974. I will remember."},
            {"role": "user", "content": "what number did i ask you to remember again?"},
        ]
        prompt = server.build_history_prompt(
            flat, "what number did i ask you to remember again?"
        )
        self.assertIn("974", prompt)
        self.assertIn("what number did i ask you to remember again?", prompt)
        self.assertIn("User:", prompt)
        self.assertIn("Assistant:", prompt)

    def test_system_messages_are_excluded(self):
        flat = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "remember the number 974"},
            {"role": "user", "content": "what number again?"},
        ]
        prompt = server.build_history_prompt(flat, "what number again?")
        self.assertNotIn("You are helpful.", prompt)
        self.assertIn("974", prompt)


class ChatMemoryTests(unittest.IsolatedAsyncioTestCase):
    @override
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store = SqliteStore(os.path.join(self.tmp_dir, "test_store.db"))
        self._orig_store = server.store
        self._orig_pool = server.pool
        self.addCleanup(setattr, server, "store", self._orig_store)
        self.addCleanup(setattr, server, "pool", self._orig_pool)
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.addCleanup(self.store.close)
        self.addCleanup(server.SESSIONS.clear)
        self.addCleanup(server._uid_cache.clear)
        server.store = self.store
        server.SESSIONS.clear()
        server._uid_cache.clear()
        self.seen: list[tuple[Any, ...]] = []

    def _pool_with(self, uid: str) -> AccountPool:
        acc = Account(index=1, cookies={"sso": "tok", "x-userid": uid}, user_id=uid)
        pool = AccountPool(os.path.join(self.tmp_dir, "accounts.txt"), store=self.store)
        pool._accounts = [acc]
        server.pool = pool
        return pool

    def _fake_run(self):
        async def fake_run(sess, prompt, **kwargs):
            self.seen.append(
                (prompt, kwargs.get("user_text"), sess.last_parent_response_id)
            )
            response_id = f"response-{len(self.seen)}"
            sess.conversation_id = sess.conversation_id or "conv-mem"
            sess.last_parent_response_id = response_id
            return TurnResult(text=response_id, response_id=response_id), []

        return fake_run

    def _msgs(self, users: list[str]) -> dict[str, Any]:
        return {"messages": [{"role": "user", "content": u} for u in users]}

    async def test_continuation_sends_only_newest_message(self):
        """Checkpoint hit -> gateway holds history; wire carries latest only."""
        self._pool_with("uid-1")
        u1 = "remember the number 974"
        u2 = "what number did i ask you to remember again?"
        with (
            patch.object(server, "run_session_turn", new=self._fake_run()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            resp = await server.chat_completions(FakeRequest(self._msgs([u1])))
            self.assertEqual(resp.status_code, 200)
            resp = await server.chat_completions(FakeRequest(self._msgs([u1, u2])))
            self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(self.seen), 2)
        prompt2, user_text2, parent2 = self.seen[1]
        self.assertEqual(prompt2, u2)
        self.assertEqual(user_text2, u2)
        self.assertEqual(parent2, "response-1")

    async def test_failover_to_new_account_resends_transcript(self):
        """Checkpoint account gone -> fresh session must not ship a bare prompt."""
        self._pool_with("uid-A")
        u1 = "remember the number 974"
        u2 = "what number did i ask you to remember again?"
        with (
            patch.object(server, "run_session_turn", new=self._fake_run()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            resp = await server.chat_completions(FakeRequest(self._msgs([u1])))
            self.assertEqual(resp.status_code, 200)
            # Account A leaves the pool; turn 2 fails over to account B.
            self._pool_with("uid-B")
            resp = await server.chat_completions(FakeRequest(self._msgs([u1, u2])))
            self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(self.seen), 2)
        prompt2, user_text2, _ = self.seen[1]
        self.assertIn("974", prompt2)
        self.assertIn(u2, prompt2)
        self.assertEqual(user_text2, u2)


class StreamUserTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_turn_forwards_latest_text_for_detection(self):

        calls: list[dict[str, Any]] = []

        class FakeSess(GrokSession):
            def __init__(self) -> None:
                super().__init__("", "", "fast")
                self.cookie_header = "ck"
                self.attachments = []
                self.last_dropped_attachment_ids = set()

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
                calls.append({"prompt": prompt, "user_text": user_text})
                yield {
                    "type": "done",
                    "result": __import__("types").SimpleNamespace(text="ok"),
                }

        transcript = "User: remember the number 974\n\nUser: what number again?"
        with patch.object(server, "refresh_statsig_pair", new=AsyncMock()):
            await server.run_session_turn(
                FakeSess(), transcript, user_text="what number again?"
            )
        self.assertEqual(calls[0]["prompt"], transcript)
        self.assertEqual(calls[0]["user_text"], "what number again?")


if __name__ == "__main__":
    unittest.main()
