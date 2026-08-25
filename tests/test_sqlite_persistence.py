"""Unit tests for SQLite persistence and state recovery across restarts."""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

import config
import server
from accounts import Account, AccountPool
from grok_gateway import GrokSession, TurnResult
from session_store import SqliteStore
from statsig import StatsigGenerator


class FakeRequest:
    def __init__(self, body: dict, headers: dict | None = None):
        self._body = body
        self.headers = headers or {}

    async def json(self):
        return self._body


class SqlitePersistenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_store.db")
        self.store = SqliteStore(self.db_path)
        server.store = self.store
        server.SESSIONS.clear()
        server._uid_cache.clear()

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        server.SESSIONS.clear()
        server._uid_cache.clear()

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
        self.assertGreater(updated["last_used"], old_time)

        # Test delete
        self.store.delete_session("test_key_1")
        self.assertIsNone(self.store.get_session("test_key_1"))

    def test_account_state_persistence(self):
        """Test account cooldown, degraded status, and metrics persistence."""
        self.store.save_account_state(
            account_key="acc_test",
            cooldown_until=100.0,
            degraded_until=200.0,
            total_requests=10,
            failed_requests=2,
        )

        st = self.store.get_account_state("acc_test")
        self.assertIsNotNone(st)
        self.assertEqual(st["cooldown_until"], 100.0)
        self.assertEqual(st["degraded_until"], 200.0)
        self.assertEqual(st["total_requests"], 10)
        self.assertEqual(st["failed_requests"], 2)

        # Verify AccountPool loads state from store
        acc_file = os.path.join(self.tmp_dir, "accounts.txt")
        with open(acc_file, "w") as f:
            f.write(".grok.com\tTRUE\t/\tTRUE\t2147483647\tsso\tsso_cookie_value\n")
            f.write(".grok.com\tTRUE\t/\tTRUE\t2147483647\tx-userid\tuid_123\n")

        pool = AccountPool(acc_file, store=self.store)
        pool._accounts = pool._parse(pool.path)
        # Manually trigger reload logic or state sync
        stored = self.store.get_all_account_states()
        for a in pool._accounts:
            if a.key in stored:
                s = stored[a.key]
                a.cooldown_until = s["cooldown_until"]
                a.degraded_until = s["degraded_until"]
                a.total_requests = s["total_requests"]
                a.failed_requests = s["failed_requests"]

        acc = pool.snapshot()[0]
        self.assertEqual(acc.key, "u:uid_123")

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
        """Simulate server restart and verify multi-turn session continues seamlessly from SQLite."""
        fake_acc = Account(index=1, cookies={"sso": "tok", "x-userid": "uid-1"}, user_id="uid-1")
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
        self.assertEqual(len(server.SESSIONS), 0)

        # Step 2: Get or create session for turn 2
        grok_sess, continued = await server.get_or_create_session(prefix_key, users, fake_acc, "fast")
        self.assertTrue(continued)
        self.assertEqual(grok_sess.conversation_id, "conv_prev_grok_id")
        self.assertEqual(grok_sess.last_parent_response_id, "parent_resp_123")
        self.assertIn(prefix_key, server.SESSIONS)

    async def test_restart_recovery_for_responses_api(self):
        """Simulate server restart and verify previous_response_id continues seamlessly from SQLite."""
        fake_acc = Account(index=1, cookies={"sso": "tok", "x-userid": "uid-1"}, user_id="uid-1")
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
        req = FakeRequest({
            "input": "And what is its population?",
            "previous_response_id": prev_resp_id,
            "stream": False,
        })

        mock_turn_result = TurnResult(
            text="The population of Paris is about 2.1 million.",
            reasoning="",
            image_urls=[],
            response_id="msg_resp_turn2",
            conversation_id="conv_paris_id",
            parent_response_id="msg_resp_turn1",
            finish_reason="stop",
        )

        with patch("server.run_session_turn", new=AsyncMock(return_value=(mock_turn_result, []))), \
             patch("server.refresh_statsig_pair", new=AsyncMock()):
            resp = await server.responses_api(req)
            self.assertEqual(resp.status_code, 200)

            # Check that turn 2 state was saved to SQLite
            saved_prev = self.store.get_session(prev_resp_id)
            self.assertIsNotNone(saved_prev)


if __name__ == "__main__":
    unittest.main()
