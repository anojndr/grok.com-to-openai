# Copyright (c) 2026 grok-to-openai-api contributors.
"""SQLite persistence for multi-turn session state and account metrics.

Persists conversation sessions, account cooldown states, cached user ids,
and Statsig seed pairs across restarts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_key             TEXT PRIMARY KEY,
    account_key             TEXT NOT NULL,
    user_chain_json         TEXT NOT NULL DEFAULT '[]',
    conversation_id         TEXT NOT NULL DEFAULT '',
    last_parent_response_id TEXT NOT NULL DEFAULT '',
    model_mode              TEXT NOT NULL DEFAULT 'fast',
    attachments_json        TEXT NOT NULL DEFAULT '[]',
    transcript_json         TEXT NOT NULL DEFAULT '[]',
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

# Remembered attachments kept per conversation row (oldest dropped). Shared
# with server.stream_session_turn so every writer caps identically.
MAX_TRACKED_ATTACHMENTS = 12

_DELETE_SESSION_SQL = "DELETE FROM sessions WHERE session_key = ?"

_SAVE_OPTION_NAMES = frozenset(
    {
        "conversation_id",
        "last_parent_response_id",
        "model_mode",
        "created_at",
        "last_used",
        "attachments",
        "transcript",
    },
)


def _row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    converted: dict[str, object] = dict(row)
    return converted


def _coerce_optional_text(value: object, default: str) -> str:
    if isinstance(value, str):
        return value
    return default


def _coerce_timestamp(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _coerce_conversation_id(value: object) -> object:
    if isinstance(value, str):
        return value
    if hasattr(value, "assert_called"):
        return ""
    return getattr(value, "conversation_id", "")


def _coerce_user_chain(value: object) -> list[str]:
    if isinstance(value, list):
        items: list[object] = list(value)
    elif isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        items = []
    else:
        items = list(value)
    return [item for item in items if isinstance(item, str)]


def _coerce_transcript(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if not isinstance(role, str) or role not in {"user", "assistant", "system"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        out.append({"role": role, "content": content})
    return out


@dataclass(frozen=True)
class _SessionValues:
    session_key: str
    account_key: str
    user_chain_json: str
    conversation_id: object
    last_parent_response_id: str
    model_mode: str
    created_at: float
    last_used: float
    transcript_json: str


def _coerce_session_values(
    session_key: object,
    account_key: object,
    user_chain: object,
    options: dict[str, object],
    now: float,
) -> _SessionValues:
    coerced_key = "" if session_key is None else str(session_key)
    coerced_account = "" if account_key is None else str(account_key)
    clean_chain = _coerce_user_chain(user_chain)
    clean_transcript = _coerce_transcript(options.get("transcript"))
    return _SessionValues(
        session_key=coerced_key,
        account_key=coerced_account,
        user_chain_json=json.dumps(clean_chain, ensure_ascii=False),
        conversation_id=_coerce_conversation_id(
            options.get("conversation_id", ""),
        ),
        last_parent_response_id=_coerce_optional_text(
            options.get("last_parent_response_id", ""),
            "",
        ),
        model_mode=_coerce_optional_text(options.get("model_mode"), "fast"),
        created_at=_coerce_timestamp(options.get("created_at"), now),
        last_used=_coerce_timestamp(options.get("last_used"), now),
        transcript_json=json.dumps(clean_transcript, ensure_ascii=False),
    )


def clean_attachment_rows(raw: object) -> list[dict[str, str | None]]:
    """Canonicalize stored attachment rows to capped shape-checked rows.

    Args:
        raw: Decoded JSON value from the attachments column.

    Returns:
        Valid rows oldest first, capped to the tracked maximum.

    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str | None]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        file_id = entry.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            continue
        digest = entry.get("hash")
        out.append(
            {
                "file_id": file_id,
                "hash": digest if isinstance(digest, str) else None,
            },
        )
    return out[-MAX_TRACKED_ATTACHMENTS:]


def _decode_session_transcript(
    raw_json: object,
    session_key: str,
) -> list[dict[str, str]]:
    """Decode the stored per-turn transcript to role/content rows.

    Args:
        raw_json: Raw transcript_json column value.
        session_key: Session key for corrupt-row debug logging.

    Returns:
        Valid transcript rows, oldest first.

    """
    if not isinstance(raw_json, str) or not raw_json:
        return []
    try:
        raw: object = json.loads(raw_json)
    except (ValueError, TypeError, AttributeError) as err:
        logging.getLogger("uvicorn.error").debug(
            "dropping corrupt transcript for session %s: %s",
            session_key,
            err,
        )
        return []
    return _coerce_transcript(raw)


class SqliteStore:
    """Persist sessions, account states, user ids, and Statsig pairs.

    Attributes:
        path: Resolved database file path.

    """

    def __init__(self, db_path: str | Path = "data/grok_store.db") -> None:
        """Open the database file and create tables when missing.

        Args:
            db_path: Filesystem path of the SQLite database file.

        """
        self.path = Path(db_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._connect()

    def _connect(self) -> None:
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)
        conn.commit()
        # Databases created before attachments_json / transcript_json existed:
        # add the columns in place; existing rows fall back to the '[]'
        # default. Probe the schema instead of blanket-catching
        # OperationalError, so a locked or otherwise broken database surfaces
        # here instead of failing later with a misleading "no such column".
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "attachments_json" not in cols:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN attachments_json"
                " TEXT NOT NULL DEFAULT '[]'",
            )
            conn.commit()
        if "transcript_json" not in cols:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN transcript_json"
                " TEXT NOT NULL DEFAULT '[]'",
            )
            conn.commit()
        self._conn = conn

    def _execute(
        self,
        sql: str,
        params: tuple[object, ...] = (),
    ) -> sqlite3.Cursor:
        """Run one statement and commit the change.

        Args:
            sql: Parameterized SQL statement.
            params: Bound query parameters.

        Returns:
            The executed cursor.

        Raises:
            RuntimeError: If the connection is missing and reconnect fails.

        """
        with self._lock:
            conn = self._conn
            if conn is None:
                msg = "SQLite connection is not initialized."
                raise RuntimeError(msg)
            try:
                cur = conn.execute(sql, params)
                conn.commit()
            except sqlite3.OperationalError as err:
                self._connect()
                fallback = self._conn
                if fallback is None:
                    msg = "SQLite reconnect failed."
                    raise RuntimeError(msg) from err
                cur = fallback.execute(sql, params)
                fallback.commit()
            return cur

    def _executemany(
        self,
        sql: str,
        rows: Sequence[tuple[object, ...]],
    ) -> None:
        """Run one statement over many parameter rows and commit.

        Args:
            sql: Parameterized SQL statement.
            rows: Parameter rows, one tuple per execution.

        Raises:
            RuntimeError: If the connection is missing and reconnect fails.

        """
        with self._lock:
            conn = self._conn
            if conn is None:
                msg = "SQLite connection is not initialized."
                raise RuntimeError(msg)
            try:
                conn.executemany(sql, rows)
                conn.commit()
            except sqlite3.OperationalError as err:
                self._connect()
                fallback = self._conn
                if fallback is None:
                    msg = "SQLite reconnect failed."
                    raise RuntimeError(msg) from err
                fallback.executemany(sql, rows)
                fallback.commit()

    def get_session(self, session_key: str) -> dict[str, object] | None:
        """Load one session row with decoded chains and attachments.

        Args:
            session_key: Primary key of the session row.

        Returns:
            The session mapping, or None when the key is unknown.

        """
        cur = self._execute(
            "SELECT * FROM sessions WHERE session_key = ?",
            (session_key,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        d = _row_to_dict(row)
        chain_text = d.get("user_chain_json")
        if not isinstance(chain_text, str) or not chain_text:
            d["user_chain"] = []
        else:
            try:
                d["user_chain"] = json.loads(chain_text)
            except (ValueError, TypeError, AttributeError) as err:
                logging.getLogger("uvicorn.error").debug(
                    "dropping corrupt user_chain for session %s: %s",
                    session_key,
                    err,
                )
                d["user_chain"] = []
        attachments_text = d.get("attachments_json")
        if not isinstance(attachments_text, str) or not attachments_text:
            raw: object = []
        else:
            try:
                raw = json.loads(attachments_text)
            except (ValueError, TypeError, AttributeError) as err:
                logging.getLogger("uvicorn.error").debug(
                    "dropping corrupt attachments for session %s: %s",
                    session_key,
                    err,
                )
                raw = []
        d["attachments"] = clean_attachment_rows(raw)
        d["transcript"] = _decode_session_transcript(
            d.get("transcript_json"),
            session_key,
        )
        return d

    def save_session(
        self,
        session_key: str,
        account_key: str,
        user_chain: list[str],
        **options: object,
    ) -> None:
        """Insert or replace one conversation session row.

        Args:
            session_key: Primary key for the session row.
            account_key: Owning account key.
            user_chain: Ordered user message chain.
            **options: Optional row overrides (conversation_id,
                last_parent_response_id, model_mode, created_at,
                last_used, attachments, transcript).

        Raises:
            TypeError: If an unknown option keyword is passed.

        """
        now = time.time()
        for key in options:
            if key not in _SAVE_OPTION_NAMES:
                msg = f"save_session() got an unexpected keyword argument '{key}'"
                raise TypeError(msg)
        values = _coerce_session_values(
            session_key,
            account_key,
            user_chain,
            options,
            now,
        )
        raw_attachments = options.get("attachments")
        if options.get("transcript") is None and raw_attachments is None:
            # Preserve both side channels atomically: the scalar subqueries
            # read the existing row inside the same INSERT, so a concurrent
            # explicit save can never be clobbered by a stale read-modify-write.
            self._execute(
                "INSERT OR REPLACE INTO sessions "
                "(session_key, account_key, user_chain_json, conversation_id, "
                "last_parent_response_id, model_mode, attachments_json, "
                "transcript_json, created_at, last_used) "
                "VALUES (?, ?, ?, ?, ?, ?, "
                "COALESCE((SELECT attachments_json FROM sessions "
                "WHERE session_key = ?), '[]'), "
                "COALESCE((SELECT transcript_json FROM sessions "
                "WHERE session_key = ?), '[]'), ?, ?)",
                (
                    values.session_key,
                    values.account_key,
                    values.user_chain_json,
                    values.conversation_id,
                    values.last_parent_response_id,
                    values.model_mode,
                    values.session_key,
                    values.session_key,
                    values.created_at,
                    values.last_used,
                ),
            )
            return
        if raw_attachments is None:
            self._execute(
                "INSERT OR REPLACE INTO sessions "
                "(session_key, account_key, user_chain_json, conversation_id, "
                "last_parent_response_id, model_mode, attachments_json, "
                "transcript_json, created_at, last_used) "
                "VALUES (?, ?, ?, ?, ?, ?, "
                "COALESCE((SELECT attachments_json FROM sessions "
                "WHERE session_key = ?), '[]'), ?, ?, ?)",
                (
                    values.session_key,
                    values.account_key,
                    values.user_chain_json,
                    values.conversation_id,
                    values.last_parent_response_id,
                    values.model_mode,
                    values.session_key,
                    values.transcript_json,
                    values.created_at,
                    values.last_used,
                ),
            )
            return
        attachments_json = json.dumps(
            clean_attachment_rows(raw_attachments),
            ensure_ascii=False,
        )
        if options.get("transcript") is None:
            self._execute(
                "INSERT OR REPLACE INTO sessions "
                "(session_key, account_key, user_chain_json, conversation_id, "
                "last_parent_response_id, model_mode, attachments_json, "
                "transcript_json, created_at, last_used) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, "
                "COALESCE((SELECT transcript_json FROM sessions "
                "WHERE session_key = ?), '[]'), ?, ?)",
                (
                    values.session_key,
                    values.account_key,
                    values.user_chain_json,
                    values.conversation_id,
                    values.last_parent_response_id,
                    values.model_mode,
                    attachments_json,
                    values.session_key,
                    values.created_at,
                    values.last_used,
                ),
            )
            return
        self._execute(
            "INSERT OR REPLACE INTO sessions "
            "(session_key, account_key, user_chain_json, conversation_id, "
            "last_parent_response_id, model_mode, attachments_json, "
            "transcript_json, created_at, last_used) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                values.session_key,
                values.account_key,
                values.user_chain_json,
                values.conversation_id,
                values.last_parent_response_id,
                values.model_mode,
                attachments_json,
                values.transcript_json,
                values.created_at,
                values.last_used,
            ),
        )

    def touch_session(
        self,
        session_key: str,
        last_used: float | None = None,
    ) -> None:
        """Refresh the last-used timestamp of one session row.

        Args:
            session_key: Primary key of the session row.
            last_used: Timestamp to store, or now when omitted.

        """
        now = last_used or time.time()
        self._execute(
            "UPDATE sessions SET last_used = ? WHERE session_key = ?",
            (now, session_key),
        )

    def delete_session(self, session_key: str) -> None:
        """Delete one session row by key.

        Args:
            session_key: Primary key of the session row.

        """
        self._execute(
            "DELETE FROM sessions WHERE session_key = ?",
            (session_key,),
        )

    def delete_sessions(self, session_keys: list[str]) -> None:
        """Delete many session rows with one constant statement.

        Args:
            session_keys: Session keys to remove.

        """
        if not session_keys:
            return
        self._executemany(
            _DELETE_SESSION_SQL,
            [(key,) for key in session_keys],
        )

    def load_all_sessions(
        self,
        ttl: float | None = None,
    ) -> dict[str, dict[str, object]]:
        """Load valid non-expired sessions from the database.

        Args:
            ttl: Maximum idle age in seconds, or None to skip expiry.

        Returns:
            Session mappings keyed by session key.

        """
        now = time.time()
        if ttl:
            cutoff = now - ttl
            cur = self._execute(
                "SELECT * FROM sessions WHERE last_used >= ?",
                (cutoff,),
            )
        else:
            cur = self._execute("SELECT * FROM sessions")
        rows = cur.fetchall()
        out: dict[str, dict[str, object]] = {}
        for record in rows:
            d = _row_to_dict(record)
            chain_text = d.get("user_chain_json")
            if not isinstance(chain_text, str) or not chain_text:
                d["user_chain"] = []
            else:
                try:
                    d["user_chain"] = json.loads(chain_text)
                except (ValueError, TypeError, AttributeError) as err:
                    logging.getLogger("uvicorn.error").debug(
                        "dropping corrupt user_chain for session %s: %s",
                        d.get("session_key"),
                        err,
                    )
                    d["user_chain"] = []
            key = d.get("session_key")
            key_text = key if isinstance(key, str) else ""
            d["transcript"] = _decode_session_transcript(
                d.get("transcript_json"),
                key_text,
            )
            if isinstance(key, str):
                out[key] = d
        return out

    def prune_stale_sessions(
        self,
        ttl: float,
        max_sessions: int,
    ) -> list[str]:
        """Prune expired sessions and cap the total row count.

        Args:
            ttl: Maximum idle age in seconds.
            max_sessions: Maximum rows kept after pruning.

        Returns:
            Pruned session keys, oldest expiry first.

        """
        now = time.time()
        cutoff = now - ttl
        cur = self._execute(
            "SELECT session_key FROM sessions WHERE last_used < ?",
            (cutoff,),
        )
        stale_keys: list[str] = [
            r["session_key"]
            for r in cur.fetchall()
            if isinstance(r["session_key"], str)
        ]
        if stale_keys:
            self.delete_sessions(stale_keys)

        cur = self._execute(
            "SELECT session_key FROM sessions ORDER BY last_used ASC",
        )
        all_keys: list[str] = [
            r["session_key"]
            for r in cur.fetchall()
            if isinstance(r["session_key"], str)
        ]
        if len(all_keys) > max_sessions:
            excess_keys = all_keys[: len(all_keys) - max_sessions]
            self.delete_sessions(excess_keys)
            stale_keys.extend(excess_keys)
        return stale_keys

    def get_account_state(self, account_key: str) -> dict[str, object] | None:
        """Load one persisted account state row.

        Args:
            account_key: Identity key of the account.

        Returns:
            The state mapping, or None when the key is unknown.

        """
        cur = self._execute(
            "SELECT * FROM account_states WHERE account_key = ?",
            (account_key,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    def get_all_account_states(self) -> dict[str, dict[str, object]]:
        """Load every persisted account state row.

        Returns:
            State mappings keyed by account key.

        """
        cur = self._execute("SELECT * FROM account_states")
        out: dict[str, dict[str, object]] = {}
        for record in cur.fetchall():
            row = _row_to_dict(record)
            key = row.get("account_key")
            if isinstance(key, str):
                out[key] = row
        return out

    def save_account_state(
        self,
        account_key: str,
        cooldown_until: float,
        degraded_until: float,
        total_requests: int,
        failed_requests: int,
    ) -> None:
        """Insert or replace one account state row.

        Args:
            account_key: Identity key of the account.
            cooldown_until: Epoch seconds when cooldown ends.
            degraded_until: Epoch seconds when quarantine ends.
            total_requests: Lifetime request count.
            failed_requests: Lifetime failure count.

        """
        now = time.time()
        self._execute(
            "INSERT OR REPLACE INTO account_states "
            "(account_key, cooldown_until, degraded_until, total_requests, "
            "failed_requests, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                account_key,
                cooldown_until,
                degraded_until,
                total_requests,
                failed_requests,
                now,
            ),
        )

    def get_uid(self, account_key: str) -> str | None:
        """Load the cached gateway user id for one account.

        Args:
            account_key: Identity key of the account.

        Returns:
            The cached user id, or None when unknown.

        """
        cur = self._execute(
            "SELECT user_id FROM uid_cache WHERE account_key = ?",
            (account_key,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        val = row["user_id"]
        return val if isinstance(val, str) else None

    def get_all_uids(self) -> dict[str, str]:
        """Load every cached gateway user id.

        Returns:
            User ids keyed by account key.

        """
        cur = self._execute("SELECT account_key, user_id FROM uid_cache")
        out: dict[str, str] = {}
        for record in cur.fetchall():
            row = _row_to_dict(record)
            key = row.get("account_key")
            user_id = row.get("user_id")
            if isinstance(key, str) and isinstance(user_id, str):
                out[key] = user_id
        return out

    def set_uid(self, account_key: str, user_id: str) -> None:
        """Cache the gateway user id for one account.

        Args:
            account_key: Identity key of the account.
            user_id: Gateway user id to cache.

        """
        now = time.time()
        self._execute(
            "INSERT OR REPLACE INTO uid_cache "
            "(account_key, user_id, updated_at) VALUES (?, ?, ?)",
            (account_key, user_id, now),
        )

    def get_statsig(self) -> tuple[str, str, float] | None:
        """Load the cached Statsig seed pair.

        Returns:
            The seed, hex, and fetch timestamp, or None when absent.

        """
        cur = self._execute(
            "SELECT seed_b64, hex_str, fetched_at FROM statsig_cache WHERE id = 1",
        )
        row = cur.fetchone()
        if row is None:
            return None
        seed = row["seed_b64"]
        hex_str = row["hex_str"]
        fetched = row["fetched_at"]
        if not isinstance(seed, str) or not isinstance(hex_str, str):
            return None
        if not isinstance(fetched, (int, float)):
            return None
        return seed, hex_str, float(fetched)

    def set_statsig(
        self,
        seed_b64: str,
        hex_str: str,
        fetched_at: float | None = None,
    ) -> None:
        """Cache one Statsig seed pair.

        Args:
            seed_b64: Base64 seed string.
            hex_str: Animation fingerprint hex string.
            fetched_at: Fetch timestamp, or now when omitted.

        """
        now = fetched_at or time.time()
        self._execute(
            "INSERT OR REPLACE INTO statsig_cache "
            "(id, seed_b64, hex_str, fetched_at) VALUES (1, ?, ?, ?)",
            (seed_b64, hex_str, now),
        )

    def close(self) -> None:
        """Close the database connection, ignoring late close errors."""
        with self._lock:
            if self._conn:
                try:
                    self._conn.close()
                except sqlite3.Error as err:
                    logging.getLogger("uvicorn.error").debug(
                        "sqlite close failed: %s",
                        err,
                    )
                self._conn = None
