"""Unit tests for SQLite persistence and state recovery across restarts."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, override
from unittest.mock import AsyncMock, patch

from fastapi import Request

import server
from accounts import Account, AccountPool
from grok_gateway import TurnResult
from session_store import SqliteStore
from statsig import StatsigGenerator


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


class SqlitePersistenceTests(unittest.IsolatedAsyncioTestCase):
    @override
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_store.db")
        self.store = SqliteStore(self.db_path)
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

    @override
    def tearDown(self):
        pass

    def test_session_crud(self):
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
        self.assertIsNotNone(sess)
        assert sess is not None
        self.assertEqual(sess["account_key"], "acc_1")
        self.assertEqual(sess["user_chain"], ["hello", "world"])
        self.assertEqual(sess["conversation_id"], "conv_123")
        self.assertEqual(sess["last_parent_response_id"], "resp_456")
        self.assertEqual(sess["model_mode"], "fast")

        # Test touch
        old_time = sess["last_used"]
        time.sleep(0.01)
        self.store.touch_session("test_key_1")
        updated = self.store.get_session("test_key_1")
        assert updated is not None
        self.assertGreater(updated["last_used"], old_time)

        # Test delete
        self.store.delete_session("test_key_1")
        self.assertIsNone(self.store.get_session("test_key_1"))

    async def test_account_state_persistence(self):
        """Test account cooldown, degraded status, and metrics persistence."""
        self.store.save_account_state(
            account_key="u:uid_123",
            cooldown_until=100.0,
            degraded_until=200.0,
            total_requests=10,
            failed_requests=2,
        )

        st = self.store.get_account_state("u:uid_123")
        self.assertIsNotNone(st)
        assert st is not None
        self.assertEqual(st["cooldown_until"], 100.0)
        self.assertEqual(st["degraded_until"], 200.0)
        self.assertEqual(st["total_requests"], 10)
        self.assertEqual(st["failed_requests"], 2)

        # Verify AccountPool loads state from store on reload
        acc_file = os.path.join(self.tmp_dir, "accounts.txt")
        await asyncio.to_thread(
            Path(acc_file).write_text,
            ".grok.com\tTRUE\t/\tTRUE\t2147483647\tsso\tsso_cookie_value\n"
            ".grok.com\tTRUE\t/\tTRUE\t2147483647\tx-userid\tuid_123\n",
        )

        pool = AccountPool(acc_file, store=self.store)
        await pool.reload_if_changed()

        acc = pool.snapshot()[0]
        self.assertEqual(acc.key, "u:uid_123")
        self.assertEqual(acc.cooldown_until, 100.0)
        self.assertEqual(acc.degraded_until, 200.0)
        self.assertEqual(acc.total_requests, 10)
        self.assertEqual(acc.failed_requests, 2)

    def test_uid_cache_persistence(self):
        """Test UID resolution caching across restarts."""
        self.store.set_uid("acc_key_1", "resolved_uid_999")
        self.assertEqual(self.store.get_uid("acc_key_1"), "resolved_uid_999")

        uids = self.store.get_all_uids()
        self.assertEqual(uids.get("acc_key_1"), "resolved_uid_999")

    def test_statsig_cache_persistence(self):
        """Test Statsig seed and animation hex persistence."""
        self.store.set_statsig("seed_base64_data", "hex_computed_value", 12345.0)

        res = self.store.get_statsig()
        self.assertIsNotNone(res)
        assert res is not None
        s_b64, h_val, f_at = res
        self.assertEqual(s_b64, "seed_base64_data")
        self.assertEqual(h_val, "hex_computed_value")
        self.assertEqual(f_at, 12345.0)

        # Verify StatsigGenerator loads cached values on init
        sg = StatsigGenerator(store=self.store)
        self.assertEqual(sg._seed_b64, "seed_base64_data")
        self.assertEqual(sg._hex, "hex_computed_value")
        self.assertTrue(sg.ready)

    async def test_restart_recovery_for_chat_completions(self):
        """Simulate server restart and verify multi-turn session continues seamlessly through public endpoint."""
        fake_acc = Account(
            index=1, cookies={"sso": "tok", "x-userid": "uid-1"}, user_id="uid-1"
        )
        pool = AccountPool(os.path.join(self.tmp_dir, "accounts.txt"), store=self.store)
        pool._accounts = [fake_acc]
        server.pool = pool

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

        async def fake_run_turn(sess, *args, **kwargs):
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
                    }
                )
            )
            self.assertEqual(resp.status_code, 200)

        # Verify turn 2 was persisted to SQLite under the full chain key
        full_key = server.chain_key(users)
        saved = self.store.get_session(full_key)
        self.assertIsNotNone(saved)
        assert saved is not None
        self.assertEqual(saved["conversation_id"], "conv_prev_grok_id")
        self.assertEqual(saved["last_parent_response_id"], "resp_turn_2")
        # Verify parent checkpoint retained its earlier parent response id
        parent_saved = self.store.get_session(prefix_key)
        assert parent_saved is not None
        self.assertEqual(parent_saved["last_parent_response_id"], "parent_resp_123")

    async def test_restart_recovery_for_responses_api(self):
        """Simulate server restart and verify previous_response_id continues seamlessly from SQLite."""
        fake_acc = Account(
            index=1, cookies={"sso": "tok", "x-userid": "uid-1"}, user_id="uid-1"
        )
        pool = AccountPool(os.path.join(self.tmp_dir, "accounts.txt"), store=self.store)
        pool._accounts = [fake_acc]
        server.pool = pool

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
            }
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

        async def fake_run_turn(sess, *args, **kwargs):
            sess.last_parent_response_id = mock_turn_result.response_id
            return mock_turn_result, []

        with (
            patch("server.run_session_turn", new=fake_run_turn),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
        ):
            resp = await server.responses_api(req)
            self.assertEqual(resp.status_code, 200)
            data = json.loads(resp.body)
            new_rid = data["id"]

            # Check that turn 2 state was saved under new_rid in SQLite
            saved_turn2 = self.store.get_session(new_rid)
            self.assertIsNotNone(saved_turn2)
            assert saved_turn2 is not None
            self.assertEqual(saved_turn2["last_parent_response_id"], "msg_resp_turn2")
            self.assertEqual(saved_turn2["conversation_id"], "conv_paris_id")

            # Check that prev_resp_id still holds the turn 1 checkpoint
            saved_prev = self.store.get_session(prev_resp_id)
            self.assertIsNotNone(saved_prev)
            assert saved_prev is not None
            self.assertEqual(saved_prev["last_parent_response_id"], "msg_resp_turn1")

    async def test_branch_uses_parent_checkpoint_not_later_sibling(self):
        """A reply to an earlier message must not inherit a later sibling turn, both in-memory and on cold-start SQLite restore."""
        fake_acc = Account(
            index=1, cookies={"sso": "tok", "x-userid": "uid-1"}, user_id="uid-1"
        )
        pool = AccountPool(os.path.join(self.tmp_dir, "accounts.txt"), store=self.store)
        pool._accounts = [fake_acc]
        server.pool = pool
        server.SESSIONS.clear()
        seen_parents = []

        u1 = "My name is Tafera. Answer in one sentence only."
        u2 = "what is my name again? Answer in one sentence only."
        u3 = 'remember the string `JB##FPrcGT2G%rY2aB@3`. reply with "understood" only.'
        u4 = "what string did i ask you to remember again? Answer in one sentence only."
        u5 = "what was my name again after cold start?"

        async def fake_run(sess, prompt, **kwargs):
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
                resp = await server.chat_completions(
                    FakeRequest(
                        {
                            "messages": [
                                {"role": "user", "content": user} for user in users
                            ]
                        }
                    )
                )
                self.assertEqual(resp.status_code, 200)

            # Turn 4: in-memory branch from Turn 2 [u1, u2]
            resp = await server.chat_completions(
                FakeRequest(
                    {
                        "messages": [
                            {"role": "user", "content": user} for user in [u1, u2, u4]
                        ]
                    }
                )
            )
            self.assertEqual(resp.status_code, 200)

            # Turn 5: cold-start branch from Turn 2 [u1, u2] after clearing in-memory SESSIONS
            server.SESSIONS.clear()
            resp = await server.chat_completions(
                FakeRequest(
                    {
                        "messages": [
                            {"role": "user", "content": user} for user in [u1, u2, u5]
                        ]
                    }
                )
            )
            self.assertEqual(resp.status_code, 200)

        # Assert correct parent response IDs seen by fake_run for each prompt
        self.assertEqual(
            seen_parents,
            [
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
            ],
        )

        parent_key = server.chain_key([u1, u2])
        sibling1_key = server.chain_key([u1, u2, u3])
        sibling2_key = server.chain_key([u1, u2, u4])
        sibling3_key = server.chain_key([u1, u2, u5])

        # Assert SQLite checkpoints for parent and all sibling branches are distinct and immutable
        parent_sess = self.store.get_session(parent_key)
        assert parent_sess is not None
        self.assertEqual(parent_sess["last_parent_response_id"], "response-2")
        sibling1_sess = self.store.get_session(sibling1_key)
        assert sibling1_sess is not None
        self.assertEqual(sibling1_sess["last_parent_response_id"], "response-3")
        sibling2_sess = self.store.get_session(sibling2_key)
        assert sibling2_sess is not None
        self.assertEqual(sibling2_sess["last_parent_response_id"], "response-4")
        sibling3_sess = self.store.get_session(sibling3_key)
        assert sibling3_sess is not None
        self.assertEqual(sibling3_sess["last_parent_response_id"], "response-5")


if __name__ == "__main__":
    unittest.main()
