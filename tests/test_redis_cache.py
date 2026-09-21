# Copyright (c) 2026 grok-to-openai-api contributors.
"""Unit tests for the Redis read-through/write-through cache.

Run: python3 -m unittest -v tests.test_redis_cache
"""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

from redis.exceptions import RedisError

import server
from accounts import AccountPool
from redis_store import HybridStore, RedisCache

if TYPE_CHECKING:
    import redis

_FAIL_MESSAGE = "boom"
_TOTAL_REQUESTS = 3


class FakePipeline:
    """Minimal non-transactional pipeline recording deletes."""

    def __init__(self, client: FakeRedis) -> None:
        """Retain the owning fake client.

        Args:
            client: Fake client receiving deletes on execute.

        """
        self._client = client
        self.keys: list[str] = []

    def delete(self, *keys: str) -> FakePipeline:
        """Record keys for deletion on execute.

        Args:
            *keys: Cache keys to delete.

        Returns:
            Self for chaining.

        """
        self.keys.extend(keys)
        return self

    def execute(self) -> list[int]:
        """Apply recorded deletes to the fake backing store.

        Returns:
            One acknowledgement per deleted key.

        """
        self._client.deleted.extend(self.keys)
        for key in self.keys:
            self._client.data.pop(key, None)
        return [1] * len(self.keys)


class FakeRedis:
    """In-memory stand-in for the redis-py client surface we use."""

    def __init__(self) -> None:
        """Create empty backing storage in serving mode."""
        self.data: dict[str, str] = {}
        self.deleted: list[str] = []
        self.fail = False

    def _maybe_fail(self) -> None:
        """Raise when failure mode is on.

        Raises:
            RedisError: When failure mode is on.

        """
        if self.fail:
            raise RedisError(_FAIL_MESSAGE)

    def ping(self) -> bool:
        """Answer PING unless failure mode is on.

        Returns:
            True when serving.

        """
        self._maybe_fail()
        return True

    def get(self, key: str) -> str | None:
        """Fetch one string value unless failure mode is on.

        Args:
            key: Cache key.

        Returns:
            The value, or None when absent.

        """
        self._maybe_fail()
        return self.data.get(key)

    def set(self, key: str, value: str, **kwargs: object) -> bool:
        """Store one string value unless failure mode is on.

        Args:
            key: Cache key.
            value: Value to store.
            **kwargs: Ignored options (e.g. ex) for signature parity.

        Returns:
            True when stored.

        """
        _ = kwargs
        self._maybe_fail()
        self.data[key] = value
        return True

    def pipeline(self, *, transaction: bool = False) -> FakePipeline:
        """Return a recording pipeline unless failure mode is on.

        Args:
            transaction: Ignored; deletes always run non-transactionally.

        Returns:
            Recording pipeline.

        """
        _ = transaction
        self._maybe_fail()
        return FakePipeline(self)

    def info(self) -> dict[str, Any]:
        """Return canned INFO fields unless failure mode is on.

        Returns:
            Memory, client, and throughput fields.

        """
        self._maybe_fail()
        return {
            "used_memory_human": "1.0M",
            "connected_clients": 2,
            "instantaneous_ops_per_sec": 10,
            "keyspace_hits": 8,
            "keyspace_misses": 2,
        }

    def close(self) -> None:
        """No-op close for the fake."""


def _as_client(fake: FakeRedis) -> redis.Redis:
    """View the fake as a redis client for cache injection.

    Args:
        fake: In-memory fake.

    Returns:
        Fake viewed as a redis client.

    """
    return cast("redis.Redis", fake)


class RedisCacheTests(unittest.IsolatedAsyncioTestCase):
    """Cover HybridStore read-through, write-through, and fail-closed paths."""

    @override
    def setUp(self) -> None:
        """Prepare an isolated hybrid store backed by fake Redis."""
        self.tmp_dir = tempfile.mkdtemp()
        self.fake = FakeRedis()
        self.store = HybridStore(
            Path(self.tmp_dir) / "t.db",
            redis_client=_as_client(self.fake),
            session_ttl=3600.0,
        )
        self.addCleanup(self.store.close)
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def _peer_store(self, name: str) -> HybridStore:
        """Open a second store sharing the fake but with an empty database.

        A read served by the peer must come from Redis: its SQLite file
        starts empty. This simulates a second process or a restart.

        Args:
            name: Database filename for the peer store.

        Returns:
            Peer store sharing the fake Redis.

        """
        peer = HybridStore(
            Path(self.tmp_dir) / name,
            redis_client=_as_client(self.fake),
            session_ttl=3600.0,
        )
        self.addCleanup(peer.close)
        return peer

    def _save_session(self, key: str = "sess-1") -> None:
        """Save one canned session row.

        Args:
            key: Session key to save.

        """
        now = time.time()
        self.store.save_session(
            session_key=key,
            account_key="u:1",
            user_chain=["hello"],
            conversation_id="c1",
            last_parent_response_id="p1",
            model_mode="fast",
            attachments=[],
            transcript=[{"role": "user", "content": "hello"}],
            created_at=now,
            last_used=now,
        )

    def test_session_write_through_then_shared_hit(self) -> None:
        """Saving mirrors to Redis; a peer with empty SQLite still serves it."""
        self._save_session()
        if "g2o:session:sess-1" not in self.fake.data:
            self.fail("session missing from redis after save")
        peer = self._peer_store("peer.db")
        row = peer.get_session("sess-1")
        if row is None or row.get("conversation_id") != "c1":
            self.fail(f"shared cache hit returned wrong row: {row!r}")
        if row.get("user_chain") != ["hello"]:
            self.fail(f"shared cache hit lost user chain: {row!r}")
        cache = peer.redis
        if cache is None or cache.hits < 1:
            self.fail("shared hit did not record a redis hit")

    def test_touch_invalidates_and_repopulates(self) -> None:
        """Touch drops the cached row; the next read repopulates from SQLite."""
        self._save_session()
        self.store.touch_session("sess-1", time.time() + 5)
        if "g2o:session:sess-1" in self.fake.data:
            self.fail("touch did not invalidate cached session")
        row = self.store.get_session("sess-1")
        if row is None:
            self.fail("session missing after touch")
        if "g2o:session:sess-1" not in self.fake.data:
            self.fail("read did not repopulate cache")

    def test_delete_propagates_to_both_layers(self) -> None:
        """Deletes remove the SQLite row and the cached key in one pipeline."""
        self._save_session()
        self.store.delete_session("sess-1")
        if "g2o:session:sess-1" in self.fake.data:
            self.fail("delete did not evict cached session")
        if self.store.get_session("sess-1") is not None:
            self.fail("session survived delete")

    def test_corrupt_cache_evicted_and_sqlite_served(self) -> None:
        """A corrupt cached payload is evicted and SQLite truth is served."""
        self._save_session()
        self.fake.data["g2o:session:sess-1"] = "not-json{{{"
        row = self.store.get_session("sess-1")
        if row is None or row.get("conversation_id") != "c1":
            self.fail(f"fallback served wrong row: {row!r}")
        if not self.fake.data.get("g2o:session:sess-1", "").startswith("{"):
            self.fail("corrupt entry was not replaced")

    def test_redis_outage_serves_sqlite(self) -> None:
        """Total Redis failure still serves SQLite and records the error."""
        self._save_session()
        self.fake.fail = True
        try:
            row = self.store.get_session("sess-1")
        finally:
            self.fake.fail = False
        if row is None or row.get("conversation_id") != "c1":
            self.fail(f"outage fallback failed: {row!r}")
        cache = self.store.redis
        if cache is None or cache.errors < 1:
            self.fail("outage did not record an error")

    def test_account_uid_statsig_round_trip(self) -> None:
        """Account states, uids, and the statsig pair mirror through Redis."""
        now = time.time()
        self.store.save_account_state("u:1", now + 10, 0.0, _TOTAL_REQUESTS, 1)
        self.store.set_uid("u:1", "resolved_uid_9")
        self.store.set_statsig("seed_b64_x", "hex_y", now)
        for key in ("g2o:account:u:1", "g2o:uid:u:1", "g2o:statsig:pair"):
            if key not in self.fake.data:
                self.fail(f"{key} missing from redis")
        peer = self._peer_store("peer.db")
        if peer.get_uid("u:1") != "resolved_uid_9":
            self.fail("uid missing from shared cache")
        pair = peer.get_statsig()
        if pair is None or pair[0] != "seed_b64_x":
            self.fail(f"statsig missing from shared cache: {pair!r}")
        state = peer.get_account_state("u:1")
        if state is None or state.get("total_requests") != _TOTAL_REQUESTS:
            self.fail(f"account state missing from shared cache: {state!r}")

    def test_redis_status_reports_observability(self) -> None:
        """Status carries counters plus INFO memory/client fields."""
        status = self.store.redis_status()
        if status.get("enabled") is not True or status.get("ok") is not True:
            self.fail(f"redis status not ok: {status!r}")
        if "used_memory_human" not in status or "keyspace_hit_ratio" not in status:
            self.fail(f"INFO fields missing: {status!r}")

    def test_disabled_store_reports_disabled(self) -> None:
        """No URL means no mirror and a disabled status shape."""
        plain = HybridStore(Path(self.tmp_dir) / "plain.db")
        self.addCleanup(plain.close)
        if plain.redis is not None:
            self.fail("mirror should be off without a URL")
        if plain.redis_status() != {"enabled": False}:
            self.fail(f"wrong disabled shape: {plain.redis_status()!r}")

    def test_key_naming_convention(self) -> None:
        """Cache keys stay lowercase colon-separated under one prefix."""
        cache = RedisCache(client=_as_client(self.fake))
        keys = [
            cache.session_key("abc"),
            cache.account_key("u:1"),
            cache.uid_key("u:1"),
            cache.statsig_key(),
        ]
        for key in keys:
            if key != key.lower() or ":" not in key or " " in key:
                self.fail(f"bad key name: {key!r}")
            if not key.startswith("g2o:"):
                self.fail(f"key missing prefix: {key!r}")


class RedisHealthTests(unittest.IsolatedAsyncioTestCase):
    """Cover the /healthz redis payload wiring."""

    @override
    def setUp(self) -> None:
        """Install an isolated hybrid store and account pool."""
        self.tmp_dir = tempfile.mkdtemp()
        self.fake = FakeRedis()
        self.store = HybridStore(
            Path(self.tmp_dir) / "t.db",
            redis_client=_as_client(self.fake),
        )
        self._orig_store = server.store
        self._orig_pool = server.pool
        server.store = self.store
        pool = AccountPool(Path(self.tmp_dir) / "accounts.txt", store=self.store)
        pool.replace_accounts([])
        server.pool = pool
        self.addCleanup(setattr, server, "store", self._orig_store)
        self.addCleanup(setattr, server, "pool", self._orig_pool)
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.addCleanup(self.store.close)
        self.addCleanup(server.SESSIONS.clear)

    async def test_healthz_includes_redis(self) -> None:
        """Health payload carries the live redis status mapping."""
        payload = await server.healthz()
        status = payload.get("redis")
        if not isinstance(status, dict) or status.get("enabled") is not True:
            self.fail(f"healthz missing redis status: {payload!r}")


if __name__ == "__main__":
    unittest.main()
