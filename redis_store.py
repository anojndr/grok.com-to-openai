# Copyright (c) 2026 grok-to-openai-api contributors.
"""Redis read-through/write-through cache over the SQLite store.

SQLite stays the source of truth; Redis mirrors hot rows (sessions, account
states, uid cache, Statsig pair) so restarts and multi-process deployments
share state without disk reads. Every Redis failure degrades to SQLite: no
cache error ever raises into request paths.

Key model (colon-separated, ``g2o:`` prefix):

    g2o:session:{session_key}  JSON session row (TTL = session TTL)
    g2o:account:{account_key}  JSON account state (no TTL; fields govern)
    g2o:uid:{account_key}      gateway user id string (long TTL)
    g2o:statsig:pair           JSON seed pair (pair TTL)

Connection model: one shared sync client over a single connection pool with
short connect/read timeouts; a non-transactional pipeline batches multi-key
deletes; request paths never issue KEYS/SCAN (prune deletes only keys SQLite
already reported stale). The sync client matches SqliteStore's blocking
style - callers already invoke the store from async paths backed by
localhost disk I/O.

Security: the default URL targets localhost only; auth comes from the URL
when set (``redis://:password@host:port/db``). No dangerous commands are
issued (no FLUSHALL/KEYS/CONFIG/DEBUG).

Observability: per-process hit/miss/write/error counters plus best-effort
INFO fields surface via ``status()`` and ``GET /healthz``.

"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, cast, override

import redis
from redis.exceptions import RedisError

from session_store import SqliteStore

if TYPE_CHECKING:
    from pathlib import Path

_LOG = logging.getLogger("uvicorn.error")

SESSION_SEGMENT = "session:"
ACCOUNT_SEGMENT = "account:"
UID_SEGMENT = "uid:"
STATSIG_SEGMENT = "statsig:pair"

_DEFAULT_PREFIX = "g2o:"
_DEFAULT_SOCKET_TIMEOUT = 2.0
_DOWN_BACKOFF_SECONDS = 30.0
_UID_TTL_SECONDS = 30 * 24 * 3600
_STATSIG_TTL_SECONDS = 1800


class RedisCache:
    """Best-effort Redis client with a circuit breaker and counters.

    Attributes:
        prefix: Key prefix applied to every cache key.
        hits: Cache hits served since process start (approximate).
        misses: Cache misses since process start (approximate).
        writes: Successful writes since process start (approximate).
        errors: Redis failures absorbed since process start (approximate).

    """

    def __init__(
        self,
        url: str = "",
        prefix: str = _DEFAULT_PREFIX,
        socket_timeout: float = _DEFAULT_SOCKET_TIMEOUT,
        client: redis.Redis | None = None,
    ) -> None:
        """Create a cache over one shared connection pool.

        Args:
            url: Redis URL (``redis://127.0.0.1:6379/0``). Ignored when
                ``client`` is injected.
            prefix: Key prefix applied to every cache key.
            socket_timeout: Connect/read timeout in seconds; cache calls
                fail fast to SQLite past it.
            client: Injected client for tests; otherwise built from ``url``.

        """
        self.prefix = prefix
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.errors = 0
        self._reachable = False
        self._down_until = 0.0
        self._client: redis.Redis | None
        if client is not None:
            self._client = client
        elif url:
            self._client = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=socket_timeout,
                socket_timeout=socket_timeout,
            )
        else:
            self._client = None

    def session_key(self, session_key: str) -> str:
        """Build the session cache key for one session key.

        Args:
            session_key: Session row key.

        Returns:
            Prefixed cache key.

        """
        return f"{self.prefix}{SESSION_SEGMENT}{session_key}"

    def account_key(self, account_key: str) -> str:
        """Build the account-state cache key for one account key.

        Args:
            account_key: Account identity key.

        Returns:
            Prefixed cache key.

        """
        return f"{self.prefix}{ACCOUNT_SEGMENT}{account_key}"

    def uid_key(self, account_key: str) -> str:
        """Build the uid cache key for one account key.

        Args:
            account_key: Account identity key.

        Returns:
            Prefixed cache key.

        """
        return f"{self.prefix}{UID_SEGMENT}{account_key}"

    def statsig_key(self) -> str:
        """Build the Statsig pair cache key.

        Returns:
            Prefixed cache key.

        """
        return f"{self.prefix}{STATSIG_SEGMENT}"

    def _live(self) -> redis.Redis | None:
        """Return the client unless the breaker is open.

        Returns:
            The client, or None while backed off after failures.

        """
        if self._client is None or time.time() < self._down_until:
            return None
        return self._client

    def _mark_down(self) -> None:
        """Record one absorbed failure and open the breaker briefly."""
        self.errors += 1
        self._reachable = False
        self._down_until = time.time() + _DOWN_BACKOFF_SECONDS

    def _mark_up(self) -> None:
        """Record one success and close the breaker."""
        self._reachable = True
        self._down_until = 0.0

    def ping_ok(self) -> bool:
        """Probe reachability without touching hit/miss counters.

        Returns:
            True when the server answered PING.

        """
        client = self._live()
        if client is None:
            return False
        try:
            client.ping()
        except RedisError as err:
            _LOG.debug("redis ping failed: %s", err)
            self._mark_down()
            return False
        self._mark_up()
        return True

    def get_json(self, key: str) -> dict[str, Any] | None:
        """Fetch one JSON object, evicting corrupt entries.

        Args:
            key: Cache key.

        Returns:
            The decoded object on a hit, else None (miss, outage, or
            corrupt entry, which is evicted best-effort).

        """
        client = self._live()
        if client is None:
            return None
        try:
            raw = client.get(key)
        except RedisError as err:
            _LOG.debug("redis GET %s failed: %s", key, err)
            self._mark_down()
            return None
        self._mark_up()
        if not isinstance(raw, str):
            self.misses += 1
            return None
        try:
            parsed: object = json.loads(raw)
        except ValueError:
            self.misses += 1
            self.delete_many([key])
            return None
        if not isinstance(parsed, dict):
            self.misses += 1
            self.delete_many([key])
            return None
        self.hits += 1
        return cast("dict[str, Any]", parsed)

    def set_json(
        self,
        key: str,
        payload: dict[str, Any],
        ttl: float | None = None,
    ) -> bool:
        """Store one JSON object with an optional TTL.

        Args:
            key: Cache key.
            payload: JSON-serializable object.
            ttl: Expiry in seconds, or None for no expiry.

        Returns:
            True when the write landed.

        """
        try:
            text = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError) as err:
            _LOG.debug("redis payload for %s not serializable: %s", key, err)
            self.errors += 1
            return False
        return self.set_text(key, text, ttl=ttl)

    def get_text(self, key: str) -> str | None:
        """Fetch one string value.

        Args:
            key: Cache key.

        Returns:
            The value on a hit, else None.

        """
        client = self._live()
        if client is None:
            return None
        try:
            raw = client.get(key)
        except RedisError as err:
            _LOG.debug("redis GET %s failed: %s", key, err)
            self._mark_down()
            return None
        self._mark_up()
        if not isinstance(raw, str):
            self.misses += 1
            return None
        self.hits += 1
        return raw

    def set_text(self, key: str, value: str, ttl: float | None = None) -> bool:
        """Store one string value with an optional TTL.

        Args:
            key: Cache key.
            value: Value to store.
            ttl: Expiry in seconds, or None for no expiry.

        Returns:
            True when the write landed.

        """
        client = self._live()
        if client is None:
            return False
        try:
            if ttl is not None:
                client.set(key, value, ex=int(ttl))
            else:
                client.set(key, value)
        except RedisError as err:
            _LOG.debug("redis SET %s failed: %s", key, err)
            self._mark_down()
            return False
        self._mark_up()
        self.writes += 1
        return True

    def delete_many(self, keys: list[str]) -> None:
        """Delete keys in one non-transactional pipeline, best-effort.

        All cache keys share the ``g2o:`` prefix pattern (not a cluster hash
        tag): a future Cluster move must tag per-entity keys (e.g.
        ``g2o:{sess}:...``) before issuing multi-key DELs.

        Args:
            keys: Cache keys to delete.

        """
        if not keys:
            return
        client = self._live()
        if client is None:
            return
        try:
            pipe = client.pipeline(transaction=False)
            pipe.delete(*keys)
            pipe.execute()
        except RedisError as err:
            _LOG.debug("redis DEL failed: %s", err)
            self._mark_down()
            return
        self._mark_up()

    def info_snapshot(self) -> dict[str, object]:
        """Capture server INFO fields for health reporting.

        Returns:
            Memory/client/throughput fields, or {} when unreachable.

        """
        client = self._live()
        if client is None:
            return {}
        try:
            raw_info: object = client.info()
        except RedisError as err:
            _LOG.debug("redis INFO failed: %s", err)
            self._mark_down()
            return {}
        self._mark_up()
        if not isinstance(raw_info, dict):
            return {}
        snapshot: dict[str, object] = {}
        for field in (
            "used_memory_human",
            "connected_clients",
            "instantaneous_ops_per_sec",
        ):
            value = raw_info.get(field)
            if value is not None:
                snapshot[field] = value
        hits = raw_info.get("keyspace_hits")
        misses = raw_info.get("keyspace_misses")
        if isinstance(hits, (int, float)) and isinstance(misses, (int, float)):
            total = hits + misses
            snapshot["keyspace_hit_ratio"] = (hits / total) if total else 1.0
        return snapshot

    def status(self) -> dict[str, object]:
        """Report cache state plus a live INFO snapshot.

        Returns:
            Status mapping for /healthz.

        """
        snapshot = self.info_snapshot()
        return {
            "enabled": True,
            "ok": self._reachable,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
            **snapshot,
        }

    def close(self) -> None:
        """Release pool connections, ignoring late close errors."""
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            client.close()
        except RedisError as err:
            _LOG.debug("redis close failed: %s", err)


def _payload_from_row(row: dict[str, object]) -> dict[str, Any]:
    """Project a decoded SQLite session row to its cache shape.

    Args:
        row: Decoded session mapping from SqliteStore.get_session.

    Returns:
        The nine fields HybridStore.get_session serves from cache.

    """
    return {
        "account_key": row.get("account_key", ""),
        "user_chain": row.get("user_chain", []),
        "conversation_id": row.get("conversation_id", ""),
        "last_parent_response_id": row.get("last_parent_response_id", ""),
        "model_mode": row.get("model_mode", "fast"),
        "attachments": row.get("attachments", []),
        "transcript": row.get("transcript", []),
        "created_at": row.get("created_at", 0.0),
        "last_used": row.get("last_used", 0.0),
    }


def _payload_from_row_account(row: dict[str, object]) -> dict[str, Any]:
    """Project a decoded SQLite account row to its cache shape.

    Args:
        row: Account-state mapping from SqliteStore.get_account_state.

    Returns:
        The five numeric fields the cache serves.

    """
    return {
        "cooldown_until": row.get("cooldown_until", 0.0),
        "degraded_until": row.get("degraded_until", 0.0),
        "total_requests": row.get("total_requests", 0),
        "failed_requests": row.get("failed_requests", 0),
        "updated_at": row.get("updated_at", 0.0),
    }


def _clean_session_row(
    session_key: str,
    cached: dict[str, Any],
) -> dict[str, object] | None:
    """Validate a cached session payload to a servable row.

    Args:
        session_key: Session key the payload was stored under.
        cached: Decoded JSON object from Redis.

    Returns:
        Row mapping shaped like SqliteStore.get_session output, or None
        when the payload fails shape checks (caller evicts and falls back).

    """
    account_key = cached.get("account_key")
    user_chain = cached.get("user_chain")
    scalars = (
        cached.get("conversation_id"),
        cached.get("last_parent_response_id"),
        cached.get("model_mode"),
    )
    lists = (cached.get("attachments"), cached.get("transcript"))
    stamps = (cached.get("created_at"), cached.get("last_used"))
    if (
        not isinstance(account_key, str)
        or not all(isinstance(text, str) for text in scalars)
        or not all(isinstance(rows, list) for rows in lists)
        or not all(
            isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
            for stamp in stamps
        )
    ):
        return None
    if not isinstance(user_chain, list) or not all(
        isinstance(item, str) for item in user_chain
    ):
        return None
    conversation_id, last_parent, model_mode = scalars
    attachments, transcript = lists
    created_at, last_used = stamps
    return {
        "session_key": session_key,
        "account_key": account_key,
        "user_chain": user_chain,
        "conversation_id": conversation_id,
        "last_parent_response_id": last_parent,
        "model_mode": model_mode,
        "attachments": attachments,
        "transcript": transcript,
        "created_at": created_at,
        "last_used": last_used,
    }


def _clean_account_row(
    account_key: str,
    cached: dict[str, Any],
) -> dict[str, object] | None:
    """Validate a cached account-state payload.

    Args:
        account_key: Account key the payload was stored under.
        cached: Decoded JSON object from Redis.

    Returns:
        State mapping shaped like SqliteStore.get_account_state output, or
        None when the payload fails shape checks.

    """
    values: dict[str, object] = {"account_key": account_key}
    for field in (
        "cooldown_until",
        "degraded_until",
        "total_requests",
        "failed_requests",
        "updated_at",
    ):
        value = cached.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        values[field] = value
    return values


class HybridStore(SqliteStore):
    """SqliteStore with a best-effort Redis mirror in front.

    Writes go to SQLite first (source of truth), then to Redis; reads hit
    Redis first and fall back to SQLite, repopulating the cache. Touches
    invalidate rather than rewrite: last_used freshness lives in SQLite and
    the next read repopulates. Bulk listing (``get_all_account_states``)
    stays on SQLite truth - the reload path runs rarely and repopulates via
    the write path anyway.

    """

    def __init__(
        self,
        db_path: str | Path = "data/grok_store.db",
        *,
        redis_url: str | None = None,
        redis_enabled: bool = True,
        session_ttl: float = 3600.0,
        redis_client: redis.Redis | None = None,
    ) -> None:
        """Open SQLite and attach the Redis mirror when configured.

        Args:
            db_path: Filesystem path of the SQLite database file.
            redis_url: Redis URL; no mirror when empty.
            redis_enabled: Master switch for the mirror.
            session_ttl: Expiry for cached session rows, in seconds.
            redis_client: Injected client for tests (uses ``redis_url``
                pool otherwise).

        """
        super().__init__(db_path)
        self.session_ttl = float(session_ttl)
        self.redis: RedisCache | None = None
        if redis_enabled and (redis_url or redis_client is not None):
            cache = RedisCache(redis_url or "", client=redis_client)
            self.redis = cache
            if cache.ping_ok():
                _LOG.info("redis cache enabled: %s", redis_url or "injected")
            else:
                _LOG.warning("redis unreachable, SQLite only until it recovers")

    def redis_status(self) -> dict[str, object]:
        """Report Redis cache state for /healthz.

        Returns:
            Live status when the mirror is on, else the disabled shape.

        """
        if self.redis is None:
            return {"enabled": False}
        return self.redis.status()

    @override
    def get_session(self, session_key: str) -> dict[str, object] | None:
        """Load one session from Redis, falling back to SQLite.

        Args:
            session_key: Primary key of the session row.

        Returns:
            The session mapping, or None when the key is unknown.

        """
        cache = self.redis
        cache_key = cache.session_key(session_key) if cache is not None else ""
        if cache is not None:
            cached = cache.get_json(cache_key)
            if cached is not None:
                row = _clean_session_row(session_key, cached)
                if row is not None:
                    return row
                cache.delete_many([cache_key])
        row = super().get_session(session_key)
        if row is not None and cache is not None:
            cache.set_json(cache_key, _payload_from_row(row), ttl=self.session_ttl)
        return row

    @override
    def save_session(
        self,
        session_key: str,
        account_key: str,
        user_chain: list[str],
        **options: object,
    ) -> None:
        """Persist one session to SQLite, then mirror to Redis.

        Args:
            session_key: Primary key for the session row.
            account_key: Owning account key.
            user_chain: Ordered user message chain.
            **options: Optional row overrides (same names as SqliteStore).

        """
        super().save_session(session_key, account_key, user_chain, **options)
        cache = self.redis
        if cache is None:
            return
        # Read back so the mirror matches SQLite exactly, including the
        # preserve-existing-row branches that merge with stored side channels.
        row = super().get_session(session_key)
        if row is None:
            return
        cache.set_json(
            cache.session_key(session_key),
            _payload_from_row(row),
            ttl=self.session_ttl,
        )

    @override
    def touch_session(
        self,
        session_key: str,
        last_used: float | None = None,
    ) -> None:
        """Refresh last-used in SQLite and invalidate the cached row.

        Args:
            session_key: Primary key of the session row.
            last_used: Timestamp to store, or now when omitted.

        """
        super().touch_session(session_key, last_used)
        if self.redis is not None:
            self.redis.delete_many([self.redis.session_key(session_key)])

    @override
    def delete_session(self, session_key: str) -> None:
        """Delete one session row from SQLite and Redis.

        Args:
            session_key: Primary key of the session row.

        """
        super().delete_session(session_key)
        if self.redis is not None:
            self.redis.delete_many([self.redis.session_key(session_key)])

    @override
    def delete_sessions(self, session_keys: list[str]) -> None:
        """Delete many session rows from SQLite and Redis.

        Args:
            session_keys: Session keys to remove.

        """
        super().delete_sessions(session_keys)
        if self.redis is not None and session_keys:
            self.redis.delete_many(
                [self.redis.session_key(key) for key in session_keys],
            )

    @override
    def get_account_state(self, account_key: str) -> dict[str, object] | None:
        """Load one account state from Redis, falling back to SQLite.

        Args:
            account_key: Identity key of the account.

        Returns:
            The state mapping, or None when the key is unknown.

        """
        cache = self.redis
        cache_key = cache.account_key(account_key) if cache is not None else ""
        if cache is not None:
            cached = cache.get_json(cache_key)
            if cached is not None:
                row = _clean_account_row(account_key, cached)
                if row is not None:
                    return row
                cache.delete_many([cache_key])
        row = super().get_account_state(account_key)
        if row is not None and cache is not None:
            cache.set_json(cache_key, _payload_from_row_account(row))
        return row

    @override
    def save_account_state(
        self,
        account_key: str,
        cooldown_until: float,
        degraded_until: float,
        total_requests: int,
        failed_requests: int,
    ) -> None:
        """Persist one account state to SQLite, then mirror to Redis.

        Args:
            account_key: Identity key of the account.
            cooldown_until: Epoch seconds when cooldown ends.
            degraded_until: Epoch seconds when quarantine ends.
            total_requests: Lifetime request count.
            failed_requests: Lifetime failure count.

        """
        super().save_account_state(
            account_key,
            cooldown_until,
            degraded_until,
            total_requests,
            failed_requests,
        )
        if self.redis is not None:
            self.redis.set_json(
                self.redis.account_key(account_key),
                {
                    "cooldown_until": cooldown_until,
                    "degraded_until": degraded_until,
                    "total_requests": total_requests,
                    "failed_requests": failed_requests,
                    "updated_at": time.time(),
                },
            )

    @override
    def get_uid(self, account_key: str) -> str | None:
        """Load one cached gateway user id, falling back to SQLite.

        Args:
            account_key: Identity key of the account.

        Returns:
            The cached user id, or None when unknown.

        """
        cache = self.redis
        if cache is not None:
            hit = cache.get_text(cache.uid_key(account_key))
            if hit is not None:
                return hit
        user_id = super().get_uid(account_key)
        if user_id is not None and cache is not None:
            cache.set_text(
                cache.uid_key(account_key),
                user_id,
                ttl=_UID_TTL_SECONDS,
            )
        return user_id

    @override
    def set_uid(self, account_key: str, user_id: str) -> None:
        """Cache one gateway user id in SQLite and Redis.

        Args:
            account_key: Identity key of the account.
            user_id: Gateway user id to cache.

        """
        super().set_uid(account_key, user_id)
        if self.redis is not None:
            self.redis.set_text(
                self.redis.uid_key(account_key),
                user_id,
                ttl=_UID_TTL_SECONDS,
            )

    @override
    def get_statsig(self) -> tuple[str, str, float] | None:
        """Load the Statsig seed pair from Redis, falling back to SQLite.

        Returns:
            The seed, hex, and fetch timestamp, or None when absent.

        """
        cache = self.redis
        if cache is not None:
            cached = cache.get_json(cache.statsig_key())
            if cached is not None:
                seed = cached.get("seed_b64")
                hex_str = cached.get("hex_str")
                fetched = cached.get("fetched_at")
                if (
                    isinstance(seed, str)
                    and isinstance(hex_str, str)
                    and isinstance(fetched, (int, float))
                    and not isinstance(fetched, bool)
                ):
                    return seed, hex_str, float(fetched)
                cache.delete_many([cache.statsig_key()])
        return super().get_statsig()

    @override
    def set_statsig(
        self,
        seed_b64: str,
        hex_str: str,
        fetched_at: float | None = None,
    ) -> None:
        """Cache one Statsig seed pair in SQLite and Redis.

        Args:
            seed_b64: Base64 seed string.
            hex_str: Animation fingerprint hex string.
            fetched_at: Fetch timestamp, or now when omitted.

        """
        super().set_statsig(seed_b64, hex_str, fetched_at)
        if self.redis is not None:
            self.redis.set_json(
                self.redis.statsig_key(),
                {
                    "seed_b64": seed_b64,
                    "hex_str": hex_str,
                    "fetched_at": fetched_at if fetched_at is not None else time.time(),
                },
                ttl=_STATSIG_TTL_SECONDS,
            )

    @override
    def close(self) -> None:
        """Close SQLite and release Redis pool connections."""
        try:
            super().close()
        finally:
            if self.redis is not None:
                self.redis.close()
                self.redis = None
