# Copyright (c) 2026 grok-to-openai-api contributors.
"""Regression tests: multi-turn conversation memory.

Incident: "remember the number 974" followed by "what number did i ask you
to remember again?" answered "3" -- total context loss. Two layers failed:

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

import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, override
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request

import server
from accounts import Account, AccountPool
from grok_gateway import GrokSession, TurnResult, default_x_grok
from session_store import SqliteStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable


HTTP_OK = 200
EXPECTED_TURNS = 2


class FakeRequest(Request):
    """Real Request carrying a canned JSON body (headers via scope)."""

    def __init__(
        self,
        body: dict[str, object],
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the fake with a body and optional headers.

        Args:
            body: Request payload returned by ``json`` and ``body``.
            headers: Optional header mapping injected via the ASGI scope.

        """
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
    async def json(self) -> dict[str, object]:
        return self._body_data

    @override
    async def body(self) -> bytes:
        return json.dumps(self._body_data).encode()


class GatewayFlagsTests(unittest.TestCase):
    """Verify gateway flags keep context without persistent memory."""

    @staticmethod
    def test_conversation_context_kept_without_persistent_memory() -> None:
        """Verify context flags stay enabled with memory disabled."""
        xgrok = default_x_grok()
        if not xgrok["keep_context"]:
            pytest.fail("keep_context not enabled")
        # Pooled accounts serve unrelated users: chats must stay temporary
        # with long-term memory off, otherwise one user's facts leak sideways.
        if not xgrok["is_temporary"]:
            pytest.fail("is_temporary not enabled")
        if not xgrok["disable_memory"]:
            pytest.fail("disable_memory not enabled")


class HistoryPromptTests(unittest.TestCase):
    """Verify history prompts preserve transcripts for fresh sessions."""

    @staticmethod
    def test_single_turn_returns_latest_byte_identical() -> None:
        """Verify a single turn returns the latest prompt unchanged."""
        flat = [{"role": "user", "content": "remember the number 974"}]
        if server.build_history_prompt(flat, "remember the number 974") != (
            "remember the number 974"
        ):
            pytest.fail("single turn prompt changed")

    @staticmethod
    def test_followup_transcript_keeps_both_user_messages() -> None:
        """Verify follow-up transcripts keep both user messages."""
        flat = [
            {"role": "user", "content": "remember the number 974"},
            {"role": "assistant", "content": "Got it, 974. I will remember."},
            {"role": "user", "content": "what number did i ask you to remember again?"},
        ]
        prompt = server.build_history_prompt(
            flat,
            "what number did i ask you to remember again?",
        )
        if "974" not in prompt:
            pytest.fail("transcript lost 974")
        if "what number did i ask you to remember again?" not in prompt:
            pytest.fail("transcript lost follow-up")
        if "User:" not in prompt:
            pytest.fail("transcript missing User label")
        if "Assistant:" not in prompt:
            pytest.fail("transcript missing Assistant label")

    @staticmethod
    def test_system_messages_are_excluded() -> None:
        """Verify system messages are excluded from transcripts."""
        flat = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "remember the number 974"},
            {"role": "user", "content": "what number again?"},
        ]
        prompt = server.build_history_prompt(flat, "what number again?")
        if "You are helpful." in prompt:
            pytest.fail("system message leaked into prompt")
        if "974" not in prompt:
            pytest.fail("transcript lost 974")


class ChatMemoryTests(unittest.IsolatedAsyncioTestCase):
    """Verify multi-turn memory across continuations and failover."""

    @override
    def setUp(self) -> None:
        """Prepare an isolated store and pool for each test."""
        tmp = Path(tempfile.mkdtemp())
        self.tmp_dir = tmp
        self.store = SqliteStore(tmp / "test_store.db")
        self._orig_store = server.store
        self._orig_pool = server.pool
        self.addCleanup(setattr, server, "store", self._orig_store)
        self.addCleanup(setattr, server, "pool", self._orig_pool)
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.addCleanup(self.store.close)
        self.addCleanup(server.SESSIONS.clear)
        self.addCleanup(server.uid_cache.clear)
        server.store = self.store
        server.SESSIONS.clear()
        server.uid_cache.clear()
        self.seen: list[tuple[str, object | None, str]] = []

    def _pool_with(self, uid: str) -> AccountPool:
        acc = Account(index=1, cookies={"sso": "tok", "x-userid": uid}, user_id=uid)
        pool = AccountPool(self.tmp_dir / "accounts.txt", store=self.store)
        pool.replace_accounts([acc])
        server.pool = pool
        return pool

    def _fake_run(
        self,
    ) -> Callable[..., Awaitable[tuple[TurnResult, list[dict[str, object]]]]]:
        async def fake_run(
            sess: GrokSession,
            prompt: str,
            **kwargs: object,
        ) -> tuple[TurnResult, list[dict[str, object]]]:
            await asyncio.sleep(0)
            self.seen.append(
                (prompt, kwargs.get("user_text"), sess.last_parent_response_id),
            )
            response_id = f"response-{len(self.seen)}"
            sess.conversation_id = sess.conversation_id or "conv-mem"
            sess.last_parent_response_id = response_id
            return TurnResult(text=response_id, response_id=response_id), []

        return fake_run

    @staticmethod
    def _msgs(users: list[str]) -> dict[str, object]:
        return {"messages": [{"role": "user", "content": u} for u in users]}

    async def test_continuation_sends_only_newest_message(self) -> None:
        """Checkpoint hit -> gateway holds history; wire carries latest only."""
        self._pool_with("uid-1")
        u1 = "remember the number 974"
        u2 = "what number did i ask you to remember again?"
        with (
            patch.object(server, "run_session_turn", new=self._fake_run()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            resp = await server.chat_completions(FakeRequest(self._msgs([u1])))
            if resp.status_code != HTTP_OK:
                pytest.fail("first turn status mismatch")
            resp = await server.chat_completions(FakeRequest(self._msgs([u1, u2])))
            if resp.status_code != HTTP_OK:
                pytest.fail("second turn status mismatch")
        if len(self.seen) != EXPECTED_TURNS:
            pytest.fail("expected two turns")
        prompt2, user_text2, parent2 = self.seen[1]
        if prompt2 != u2:
            pytest.fail("continuation prompt mismatch")
        if user_text2 != u2:
            pytest.fail("continuation user_text mismatch")
        if parent2 != "response-1":
            pytest.fail("continuation parent mismatch")

    async def test_failover_to_new_account_resends_transcript(self) -> None:
        """Checkpoint account gone -> fresh session must not ship a bare prompt."""
        self._pool_with("uid-A")
        u1 = "remember the number 974"
        u2 = "what number did i ask you to remember again?"
        with (
            patch.object(server, "run_session_turn", new=self._fake_run()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            resp = await server.chat_completions(FakeRequest(self._msgs([u1])))
            if resp.status_code != HTTP_OK:
                pytest.fail("first turn status mismatch")
            # Account A leaves the pool; turn 2 fails over to account B.
            self._pool_with("uid-B")
            resp = await server.chat_completions(FakeRequest(self._msgs([u1, u2])))
            if resp.status_code != HTTP_OK:
                pytest.fail("second turn status mismatch")
        if len(self.seen) != EXPECTED_TURNS:
            pytest.fail("expected two turns")
        prompt2, user_text2, _ = self.seen[1]
        if not isinstance(prompt2, str):
            pytest.fail("failover prompt is not text")
        if "974" not in prompt2:
            pytest.fail("failover lost 974")
        if u2 not in prompt2:
            pytest.fail("failover lost follow-up")
        if user_text2 != u2:
            pytest.fail("failover user_text mismatch")


class StreamUserTextTests(unittest.IsolatedAsyncioTestCase):
    """Verify streaming turns forward latest text for detection."""

    @staticmethod
    async def test_stream_turn_forwards_latest_text_for_detection() -> None:
        """Verify stream turns forward the latest text for detection."""
        calls: list[dict[str, object]] = []

        class FakeSess(GrokSession):
            """Fake session capturing ask arguments."""

            def __init__(self) -> None:
                """Initialize the fake with empty state."""
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
            ) -> AsyncIterator[dict[str, object]]:
                calls.append({"prompt": prompt, "user_text": user_text})
                yield {
                    "type": "done",
                    "result": SimpleNamespace(text="ok"),
                }

        transcript = "User: remember the number 974\n\nUser: what number again?"
        with patch.object(server, "refresh_statsig_pair", new=AsyncMock()):
            await server.run_session_turn(
                FakeSess(),
                transcript,
                user_text="what number again?",
            )
        if calls[0]["prompt"] != transcript:
            pytest.fail("stream prompt mismatch")
        if calls[0]["user_text"] != "what number again?":
            pytest.fail("stream user_text mismatch")


if __name__ == "__main__":
    unittest.main()
