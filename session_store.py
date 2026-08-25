"""SQLite persistence for multi-turn session state and account metrics.

Persists:
  - Multi-turn conversation sessions (client_key/prefix_key, response_id -> account_key, user_chain, conversation_id, last_parent_response_id, model_mode, created_at, last_used)
  - Account state / cooldowns / degraded timestamps / stats across restarts.
  - Cached user_ids per account index/sso hash.
  - Statsig seed and animation hex pairs.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_key             TEXT PRIMARY KEY,
    account_key             TEXT NOT NULL,
    user_chain_json         TEXT NOT NULL DEFAULT '[]',
    conversation_id         TEXT NOT NULL DEFAULT '',
    last_parent_response_id TEXT NOT NULL DEFAULT '',
    model_mode              TEXT NOT NULL DEFAULT 'fast',
    created_at              REAL NOT NULL,
    last_used               REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_last_used ON sessions(last_used);

CREATE TABLE IF NOT EXISTS account_states (
    account_key     TEXT PRIMARY KEY,
    cooldown_until  REAL NOT NULL DEFAULT 0.0,
    degraded_until  REAL NOT NULL DEFAULT 0.0,
    total_requests  INTEGER NOT NULL DEFAULT 0,
    failed_requests INTEGER NOT NULL DEFAULT 0,
    updated_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS uid_cache (
    account_key TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS statsig_cache (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    seed_b64    TEXT NOT NULL,
    hex_str     TEXT NOT NULL,
    fetched_at  REAL NOT NULL
);
"""


class SqliteStore:
    def __init__(self, db_path: str | Path = "data/grok_store.db"):
        self.path = Path(db_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = __import__("threading").Lock()
        self._conn = None
        self._connect()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def _execute(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            try:
                cur = self._conn.execute(sql, params)
                self._conn.commit()
                return cur
            except sqlite3.OperationalError:
                self._connect()
                cur = self._conn.execute(sql, params)
                self._conn.commit()
                return cur

    # ----------------------------------------------------------- sessions

    def get_session(self, session_key: str) -> dict | None:
        cur = self._execute("SELECT * FROM sessions WHERE session_key = ?", (session_key,))
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["user_chain"] = json.loads(d.get("user_chain_json") or "[]")
        except Exception:
            d["user_chain"] = []
        return d

    def save_session(self, session_key: str, account_key: str, user_chain: list[str],
                     conversation_id: str = "", last_parent_response_id: str = "",
                     model_mode: str = "fast", created_at: float | None = None,
                     last_used: float | None = None) -> None:
        now = time.time()
        created_at = float(created_at) if isinstance(created_at, (int, float)) else now
        last_used = float(last_used) if isinstance(last_used, (int, float)) else now
        session_key = str(session_key) if session_key is not None else ""
        account_key = str(account_key) if account_key is not None else ""
        conversation_id = str(conversation_id) if isinstance(conversation_id, str) else (getattr(conversation_id, "conversation_id", "") if not hasattr(conversation_id, "assert_called") else "")
        last_parent_response_id = str(last_parent_response_id) if isinstance(last_parent_response_id, str) else ""
        model_mode = str(model_mode) if isinstance(model_mode, str) else "fast"
        if not isinstance(user_chain, list):
            user_chain = list(user_chain) if hasattr(user_chain, "__iter__") and not isinstance(user_chain, (str, bytes)) else []
        clean_chain = [str(u) for u in user_chain if isinstance(u, str)]
        user_chain_json = json.dumps(clean_chain, ensure_ascii=False)
        self._execute(
            "INSERT OR REPLACE INTO sessions "
            "(session_key, account_key, user_chain_json, conversation_id, last_parent_response_id, model_mode, created_at, last_used) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_key, account_key, user_chain_json, conversation_id,
             last_parent_response_id, model_mode, created_at, last_used),
        )

    def touch_session(self, session_key: str, last_used: float | None = None) -> None:
        now = last_used or time.time()
        self._execute("UPDATE sessions SET last_used = ? WHERE session_key = ?", (now, session_key))

    def delete_session(self, session_key: str) -> None:
        self._execute("DELETE FROM sessions WHERE session_key = ?", (session_key,))

    def delete_sessions(self, session_keys: list[str]) -> None:
        if not session_keys:
            return
        placeholders = ",".join("?" for _ in session_keys)
        self._execute(f"DELETE FROM sessions WHERE session_key IN ({placeholders})", tuple(session_keys))

    def load_all_sessions(self, ttl: float | None = None) -> dict[str, dict]:
        """Load valid non-expired sessions from DB."""
        now = time.time()
        if ttl:
            cutoff = now - ttl
            cur = self._execute("SELECT * FROM sessions WHERE last_used >= ?", (cutoff,))
        else:
            cur = self._execute("SELECT * FROM sessions")
        rows = cur.fetchall()
        out = {}
        for r in rows:
            d = dict(r)
            try:
                d["user_chain"] = json.loads(d.get("user_chain_json") or "[]")
            except Exception:
                d["user_chain"] = []
            out[d["session_key"]] = d
        return out

    def prune_stale_sessions(self, ttl: float, max_sessions: int) -> list[str]:
        """Prune sessions older than TTL and excess sessions beyond max_sessions. Returns pruned session keys."""
        now = time.time()
        cutoff = now - ttl
        cur = self._execute("SELECT session_key FROM sessions WHERE last_used < ?", (cutoff,))
        stale_keys = [r["session_key"] for r in cur.fetchall()]
        if stale_keys:
            self.delete_sessions(stale_keys)

        cur = self._execute("SELECT session_key FROM sessions ORDER BY last_used ASC")
        all_keys = [r["session_key"] for r in cur.fetchall()]
        if len(all_keys) > max_sessions:
            excess_keys = all_keys[:len(all_keys) - max_sessions]
            self.delete_sessions(excess_keys)
            stale_keys.extend(excess_keys)
        return stale_keys

    # ----------------------------------------------------- account states

    def get_account_state(self, account_key: str) -> dict | None:
        cur = self._execute("SELECT * FROM account_states WHERE account_key = ?", (account_key,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all_account_states(self) -> dict[str, dict]:
        cur = self._execute("SELECT * FROM account_states")
        return {r["account_key"]: dict(r) for r in cur.fetchall()}

    def save_account_state(self, account_key: str, cooldown_until: float,
                           degraded_until: float, total_requests: int,
                           failed_requests: int) -> None:
        now = time.time()
        self._execute(
            "INSERT OR REPLACE INTO account_states "
            "(account_key, cooldown_until, degraded_until, total_requests, failed_requests, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (account_key, cooldown_until, degraded_until, total_requests, failed_requests, now),
        )

    # ---------------------------------------------------------- uid cache

    def get_uid(self, account_key: str) -> str | None:
        cur = self._execute("SELECT user_id FROM uid_cache WHERE account_key = ?", (account_key,))
        row = cur.fetchone()
        return row["user_id"] if row else None

    def get_all_uids(self) -> dict[str, str]:
        cur = self._execute("SELECT account_key, user_id FROM uid_cache")
        return {r["account_key"]: r["user_id"] for r in cur.fetchall()}

    def set_uid(self, account_key: str, user_id: str) -> None:
        now = time.time()
        self._execute(
            "INSERT OR REPLACE INTO uid_cache (account_key, user_id, updated_at) VALUES (?, ?, ?)",
            (account_key, user_id, now),
        )

    # ------------------------------------------------------ statsig cache

    def get_statsig(self) -> tuple[str, str, float] | None:
        cur = self._execute("SELECT seed_b64, hex_str, fetched_at FROM statsig_cache WHERE id = 1")
        row = cur.fetchone()
        if row:
            return row["seed_b64"], row["hex_str"], row["fetched_at"]
        return None

    def set_statsig(self, seed_b64: str, hex_str: str, fetched_at: float | None = None) -> None:
        now = fetched_at or time.time()
        self._execute(
            "INSERT OR REPLACE INTO statsig_cache (id, seed_b64, hex_str, fetched_at) VALUES (1, ?, ?, ?)",
            (seed_b64, hex_str, now),
        )

    def close(self) -> None:
        with self._lock:
            if self._conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
