# Copyright (c) 2026 grok-to-openai-api contributors.
"""Unit tests for SQLite persistence and state recovery across restarts."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, override
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request

import server
from accounts import Account, AccountPool
from grok_gateway import TurnResult
from session_store import SqliteStore
from statsig import StatsigGenerator

if TYPE_CHECKING:
    from grok_gateway import GrokSession

_HTTP_OK = 200
_COOLDOWN_UNTIL = 100.0
_DEGRADED_UNTIL = 200.0
_TOTAL_REQUESTS = 10
_FAILED_REQUESTS = 2
_FETCHED_AT = 12345.0


class FakeRequest(Request):
    """Real Request carrying a canned JSON body (headers via scope)."""

    def __init__(
        self,
        body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        """Store the canned JSON body and header map."""
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


class SqlitePersistenceTests(unittest.IsolatedAsyncioTestCase):
    """Cover SQLite persistence and restart recovery."""

    @override
    def setUp(self) -> None:
        """Prepare an isolated store and patch server globals."""
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test_store.db"
        self.store = SqliteStore(self.db_path)
        self._orig_store = server.store
        self._orig_pool = server.pool
        self.addCleanup(setattr, server, "store", self._orig_store)
        self.addCleanup(setattr, server, "pool", self._orig_pool)
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.addCleanup(self.store.close)
        self.addCleanup(server.SESSIONS.clear)
        self.addCleanup(server.uid_cache.clear)
        server.store = self.store
        server.SESSIONS.clear()
        server.uid_cache.clear()

    @override
    def tearDown(self) -> None:
        """Release per-test resources."""

    def _install_pool(self, acc: Account) -> None:
        """Install a single-account pool backed by the test store."""
        pool = AccountPool(Path(self.tmp_dir) / "accounts.txt", store=self.store)
        pool.replace_accounts([acc])
        server.pool = pool

    def _check_checkpoint(self, users: list[str], expected_parent: str) -> None:
        """Require the stored checkpoint to carry the expected parent."""
        saved = self.store.get_session(server.chain_key(users))
        if saved is None:
            pytest.fail("checkpoint missing")
        if saved["last_parent_response_id"] != expected_parent:
            pytest.fail("checkpoint parent mismatch")

    @staticmethod
    async def _post_chat_turn(users: list[str]) -> None:
        """Post one chat turn and require an HTTP 200 reply."""
        request = FakeRequest(
            {"messages": [{"role": "user", "content": user} for user in users]},
        )
        resp = await server.chat_completions(request)
        if resp.status_code != _HTTP_OK:
            pytest.fail("chat turn failed")

    async def _check_pool_reload(self, acc_file: Path) -> None:
        """Verify the pool reloads persisted account state."""
        pool = AccountPool(acc_file, store=self.store)
        await pool.reload_if_changed()

        acc = pool.snapshot()[0]
        if acc.key != "u:uid_123":
            pytest.fail("account key mismatch")
        if acc.cooldown_until != _COOLDOWN_UNTIL:
            pytest.fail("pool cooldown mismatch")
        if acc.degraded_until != _DEGRADED_UNTIL:
            pytest.fail("pool degraded mismatch")
        if acc.total_requests != _TOTAL_REQUESTS:
            pytest.fail("pool total requests mismatch")
        if acc.failed_requests != _FAILED_REQUESTS:
            pytest.fail("pool failed requests mismatch")

    @staticmethod
    def _check_session_fields(sess: dict[str, object]) -> None:
        """Require the stored session to carry the expected fields."""
        if sess["account_key"] != "acc_1":
            pytest.fail("account key mismatch")
        if sess["user_chain"] != ["hello", "world"]:
            pytest.fail("user chain mismatch")
        if sess["conversation_id"] != "conv_123":
            pytest.fail("conversation id mismatch")
        if sess["last_parent_response_id"] != "resp_456":
            pytest.fail("parent response mismatch")
        if sess["model_mode"] != "fast":
            pytest.fail("model mode mismatch")

    def test_session_crud(self) -> None:
        """Test basic session storage and retrieval in sqlite."""
        self.store.save_session(
            session_key="test_key_1",
            account_key="acc_1",
            user_chain=["hello", "world"],
            conversation_id="conv_123",
            last_parent_response_id="resp_456",
            model_mode="fast",
        )

        sess = self.store.get_session("test_key_1")
        if sess is None:
            pytest.fail("session not stored")
        self._check_session_fields(sess)

        # Test touch
        old_time = sess["last_used"]
        if not isinstance(old_time, float):
            pytest.fail("last_used is not a float")
        time.sleep(0.01)
        self.store.touch_session("test_key_1")
        updated = self.store.get_session("test_key_1")
        if updated is None:
            pytest.fail("session missing after touch")
        updated_last_used = updated["last_used"]
        if not isinstance(updated_last_used, float):
            pytest.fail("updated last_used is not a float")
        if updated_last_used <= old_time:
            pytest.fail("touch did not advance last_used")

        # Test delete
        self.store.delete_session("test_key_1")
        if self.store.get_session("test_key_1") is not None:
            pytest.fail("session survived delete")

    async def test_account_state_persistence(self) -> None:
        """Test account cooldown, degraded status, and metrics persistence."""
        self.store.save_account_state(
            account_key="u:uid_123",
            cooldown_until=_COOLDOWN_UNTIL,
            degraded_until=_DEGRADED_UNTIL,
            total_requests=_TOTAL_REQUESTS,
            failed_requests=_FAILED_REQUESTS,
        )

        st = self.store.get_account_state("u:uid_123")
        if st is None:
            pytest.fail("account state not stored")
        if st["cooldown_until"] != _COOLDOWN_UNTIL:
            pytest.fail("cooldown mismatch")
        if st["degraded_until"] != _DEGRADED_UNTIL:
            pytest.fail("degraded mismatch")
        if st["total_requests"] != _TOTAL_REQUESTS:
            pytest.fail("total requests mismatch")
        if st["failed_requests"] != _FAILED_REQUESTS:
            pytest.fail("failed requests mismatch")

        # Verify AccountPool loads state from store on reload
        acc_file = Path(self.tmp_dir) / "accounts.txt"
        await asyncio.to_thread(
            acc_file.write_text,
            ".grok.com\tTRUE\t/\tTRUE\t2147483647\tsso\tsso_cookie_value\n"
            ".grok.com\tTRUE\t/\tTRUE\t2147483647\tx-userid\tuid_123\n",
        )

        await self._check_pool_reload(acc_file)

    def test_uid_cache_persistence(self) -> None:
        """Test UID resolution caching across restarts."""
        self.store.set_uid("acc_key_1", "resolved_uid_999")
        if self.store.get_uid("acc_key_1") != "resolved_uid_999":
            pytest.fail("uid roundtrip mismatch")

        uids = self.store.get_all_uids()
        if uids.get("acc_key_1") != "resolved_uid_999":
            pytest.fail("uid cache mismatch")

    def test_statsig_cache_persistence(self) -> None:
        """Test Statsig seed and animation hex persistence."""
        self.store.set_statsig("seed_base64_data", "hex_computed_value", _FETCHED_AT)

        res = self.store.get_statsig()
        if res is None:
            pytest.fail("statsig not stored")
        s_b64, h_val, f_at = res
        if s_b64 != "seed_base64_data":
            pytest.fail("seed mismatch")
        if h_val != "hex_computed_value":
            pytest.fail("hex mismatch")
        if f_at != _FETCHED_AT:
            pytest.fail("fetched_at mismatch")

        # Verify StatsigGenerator loads cached values on init
        sg = StatsigGenerator(store=self.store)
        if sg.seed_b64 != "seed_base64_data":
            pytest.fail("generator seed mismatch")
        if sg.hex_digest != "hex_computed_value":
            pytest.fail("generator hex mismatch")
        if not sg.ready:
            pytest.fail("generator not ready")

    async def test_restart_recovery_for_chat_completions(self) -> None:
        """Simulate restart and verify chat session recovery."""
        fake_acc = Account(
            index=1,
            cookies={"sso": "tok", "x-userid": "uid-1"},
            user_id="uid-1",
        )
        self._install_pool(fake_acc)

        users = ["What is 2+2?", "And multiply by 3?"]
        prefix_key = server.chain_key(users[:-1])

        # Step 1: Pre-populate sqlite store simulating turn 1 before restart
        self.store.save_session(
            session_key=prefix_key,
            account_key=fake_acc.key,
            user_chain=["What is 2+2?"],
            conversation_id="conv_prev_grok_id",
            last_parent_response_id="parent_resp_123",
            model_mode="fast",
        )

        # In-memory SESSIONS is empty (simulating restart)
        server.SESSIONS.clear()

        # Step 2: Run turn 2 through chat completions public endpoint
        mock_turn_result = TurnResult(
            text="The result is 12.",
            response_id="resp_turn_2",
            conversation_id="conv_prev_grok_id",
            parent_response_id="parent_resp_123",
            finish_reason="stop",
        )

        async def fake_run_turn(
            sess: GrokSession,
            *_args: object,
            **_kwargs: object,
        ) -> tuple[TurnResult, list[dict[str, Any]]]:
            """Replay a canned turn result for restart recovery.

            Returns:
                The canned turn result with no follow-up events.

            """
            await asyncio.sleep(0)
            sess.last_parent_response_id = mock_turn_result.response_id
            return mock_turn_result, []

        with (
            patch("server.run_session_turn", new=fake_run_turn),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
        ):
            resp = await server.chat_completions(
                FakeRequest(
                    {
                        "messages": [{"role": "user", "content": u} for u in users],
                    },
                ),
            )
            if resp.status_code != _HTTP_OK:
                pytest.fail("chat turn failed")

        # Verify turn 2 was persisted to SQLite under the full chain key
        full_key = server.chain_key(users)
        saved = self.store.get_session(full_key)
        if saved is None:
            pytest.fail("turn not persisted")
        if saved["conversation_id"] != "conv_prev_grok_id":
            pytest.fail("conversation id mismatch")
        if saved["last_parent_response_id"] != "resp_turn_2":
            pytest.fail("parent response mismatch")
        # Verify parent checkpoint retained its earlier parent response id
        parent_saved = self.store.get_session(prefix_key)
        if parent_saved is None:
            pytest.fail("parent checkpoint missing")
        if parent_saved["last_parent_response_id"] != "parent_resp_123":
            pytest.fail("parent checkpoint overwritten")

    async def test_restart_recovery_for_responses_api(self) -> None:
        """Simulate restart and verify response id recovery."""
        fake_acc = Account(
            index=1,
            cookies={"sso": "tok", "x-userid": "uid-1"},
            user_id="uid-1",
        )
        self._install_pool(fake_acc)

        prev_resp_id = "resp_previous_12345"

        # Pre-populate sqlite store simulating turn 1 before restart
        self.store.save_session(
            session_key=prev_resp_id,
            account_key=fake_acc.key,
            user_chain=["What is the capital of France?"],
            conversation_id="conv_paris_id",
            last_parent_response_id="msg_resp_turn1",
            model_mode="fast",
        )

        # In-memory SESSIONS is empty (simulating restart)
        server.SESSIONS.clear()

        # Turn 2 using previous_response_id
        req = FakeRequest(
            {
                "input": "And what is its population?",
                "previous_response_id": prev_resp_id,
                "stream": False,
            },
        )

        mock_turn_result = TurnResult(
            text="The population of Paris is about 2.1 million.",
            reasoning="",
            image_urls=[],
            response_id="msg_resp_turn2",
            conversation_id="conv_paris_id",
            parent_response_id="msg_resp_turn1",
            finish_reason="stop",
        )

        async def fake_run_turn(
            sess: GrokSession,
            *_args: object,
            **_kwargs: object,
        ) -> tuple[TurnResult, list[dict[str, Any]]]:
            """Replay a canned turn result for restart recovery.

            Returns:
                The canned turn result with no follow-up events.

            """
            await asyncio.sleep(0)
            sess.last_parent_response_id = mock_turn_result.response_id
            return mock_turn_result, []

        with (
            patch("server.run_session_turn", new=fake_run_turn),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
        ):
            resp = await server.responses_api(req)
            if resp.status_code != _HTTP_OK:
                pytest.fail("responses turn failed")
            data = json.loads(bytes(resp.body))
            new_rid = data["id"]

            # Check that turn 2 state was saved under new_rid in SQLite
            saved_turn2 = self.store.get_session(new_rid)
            if saved_turn2 is None:
                pytest.fail("turn 2 not persisted")
            if saved_turn2["last_parent_response_id"] != "msg_resp_turn2":
                pytest.fail("turn 2 parent mismatch")
            if saved_turn2["conversation_id"] != "conv_paris_id":
                pytest.fail("turn 2 conversation mismatch")

            # Check that prev_resp_id still holds the turn 1 checkpoint
            saved_prev = self.store.get_session(prev_resp_id)
            if saved_prev is None:
                pytest.fail("turn 1 checkpoint missing")
            if saved_prev["last_parent_response_id"] != "msg_resp_turn1":
                pytest.fail("turn 1 checkpoint overwritten")

    async def test_branch_uses_parent_checkpoint_not_later_sibling(self) -> None:
        """Verify branches attach to the parent checkpoint."""
        fake_acc = Account(
            index=1,
            cookies={"sso": "tok", "x-userid": "uid-1"},
            user_id="uid-1",
        )
        self._install_pool(fake_acc)
        server.SESSIONS.clear()
        seen_parents: list[tuple[str, str]] = []

        u1 = "My name is Tafera. Answer in one sentence only."
        u2 = "what is my name again? Answer in one sentence only."
        u3 = 'remember the string `JB##FPrcGT2G%rY2aB@3`. reply with "understood" only.'
        u4 = "what string did i ask you to remember again? Answer in one sentence only."
        u5 = "what was my name again after cold start?"

        async def fake_run(
            sess: GrokSession,
            prompt: str,
            **_kwargs: object,
        ) -> tuple[TurnResult, list[dict[str, Any]]]:
            """Record parent checkpoints while replaying canned responses.

            Returns:
                The canned turn response with no follow-up events.

            """
            await asyncio.sleep(0)
            seen_parents.append((prompt, sess.last_parent_response_id))
            response_id = f"response-{len(seen_parents)}"
            sess.conversation_id = sess.conversation_id or "conv-tafera"
            sess.last_parent_response_id = response_id
            return TurnResult(text=response_id, response_id=response_id), []

        with (
            patch.object(server, "run_session_turn", new=fake_run),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            # Turns 1, 2, 3 (linear conversation)
            for users in ([u1], [u1, u2], [u1, u2, u3]):
                await self._post_chat_turn(users)

            # Turn 4: in-memory branch from Turn 2 [u1, u2]
            await self._post_chat_turn([u1, u2, u4])

            # Turn 5: cold-start branch from Turn 2 [u1, u2]
            server.SESSIONS.clear()
            await self._post_chat_turn([u1, u2, u5])

        # Assert correct parent response IDs seen by fake_run for each prompt
        if seen_parents != [
            (u1, ""),
            (u2, "response-1"),
            (u3, "response-2"),
            (
                u4,
                "response-2",
            ),  # in-memory branch correctly attached to Turn 2 output
            (
                u5,
                "response-2",
            ),  # cold-start SQLite branch correctly attached to Turn 2 output
        ]:
            pytest.fail("branch parents mismatch")

        # Assert SQLite checkpoints for parent and all sibling branches
        self._check_checkpoint([u1, u2], "response-2")
        self._check_checkpoint([u1, u2, u3], "response-3")
        self._check_checkpoint([u1, u2, u5], "response-5")

    async def test_chained_responses_keeps_repeated_tail(self) -> None:
        """Verify a repeated follow-up text persists as a new chain entry."""
        fake_acc = Account(
            index=1,
            cookies={"sso": "tok", "x-userid": "uid-1"},
            user_id="uid-1",
        )
        self._install_pool(fake_acc)
        server.SESSIONS.clear()

        async def fake_run_turn(
            sess: GrokSession,
            *_args: object,
            **_kwargs: object,
        ) -> tuple[TurnResult, list[dict[str, Any]]]:
            """Replay a canned turn result for the chained follow-up.

            Returns:
                The canned turn result with no follow-up events.

            """
            await asyncio.sleep(0)
            sess.last_parent_response_id = "msg_resp_turn2"
            return TurnResult(
                text="Sections to avoid: local studies.",
                response_id="msg_resp_turn2",
                conversation_id="conv_chain",
                parent_response_id="msg_resp_turn1",
                finish_reason="stop",
            ), []

        with (
            patch("server.run_session_turn", new=fake_run_turn),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
        ):
            first = await server.responses_api(
                FakeRequest({"input": "what sections to avoid"}),
            )
            if first.status_code != _HTTP_OK:
                pytest.fail("root turn failed")
            first_id = json.loads(bytes(first.body))["id"]
            # Repeat the exact same text as a chained follow-up: the stored
            # chain must keep both entries, not dedup the tail away.
            second = await server.responses_api(
                FakeRequest({
                    "input": "what sections to avoid",
                    "previous_response_id": first_id,
                }),
            )
            if second.status_code != _HTTP_OK:
                pytest.fail("chained turn failed")
            second_id = json.loads(bytes(second.body))["id"]

        saved = self.store.get_session(second_id)
        if saved is None:
            pytest.fail("chained turn not persisted")
        chain = saved.get("user_chain")
        if chain != ["what sections to avoid", "what sections to avoid"]:
            pytest.fail(f"repeated tail dropped from chain: {chain!r}")


if __name__ == "__main__":
    unittest.main()
