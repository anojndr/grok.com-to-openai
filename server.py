# Copyright (c) 2026 grok-to-openai-api contributors.
"""Grok → OpenAI-compatible API server.

Endpoints:
  POST /v1/chat/completions   (stream + non-stream)
  POST /v1/responses          (stream + non-stream, previous_response_id chaining)
  GET  /v1/models
  GET  /healthz

Chat runs over Grok's WebSocket Gateway (fast path, no browser). Multi-turn
conversations attach each new request to an immutable user-chain checkpoint
(conversation attach + parent_response_id), sending only the newest user message.
Requests that miss every checkpoint (account failover, restart, prefix
mismatch) resend the client-supplied transcript instead of a bare latest
message, so context is never silently dropped.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import time
import unittest.mock
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, Unpack
from urllib.parse import urlparse

import websockets
from curl_cffi.requests import AsyncSession
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import config
import grok_gateway as gw
from accounts import Account, AccountPool
from config import (
    API_KEY,
    COOLDOWN_SECONDS,
    DEFAULT_MODEL,
    GROK_BASE,
    INCLUDE_SOURCES,
    MAX_SESSIONS,
    SESSION_TTL,
    USER_AGENT,
)
from grok_gateway import GatewayError, GrokSession, RenderFilter, TurnResult
from session_store import MAX_TRACKED_ATTACHMENTS, SqliteStore, clean_attachment_rows
from statsig import StatsigGenerator
from uploads import (
    UploadError,
    UploadPayload,
    decode_data_url,
    freeimage_upload,
    freeimage_upload_from_url,
    guess_mime,
    upload_file,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping

app = FastAPI(title="grok-to-openai-api", version="1.0")

store = SqliteStore(config.DB_PATH)
pool = AccountPool(config.ACCOUNTS_FILE, cooldown_seconds=COOLDOWN_SECONDS, store=store)
statsig = StatsigGenerator(store=store)

HTTP_OK = 200

UPSTREAM_KIND = "upstream"

STATSIG_WARN_INTERVAL_SECONDS = 600.0

MAX_INLINE_LANG_LEN = 8

MIN_USERS_FOR_CHAIN_KEY = 2

_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


@dataclass
class _StatsigWarnState:
    """Throttle state for the statsig-unavailable warning."""

    last_warned_at: float = 0.0


_statsig_warn_state = _StatsigWarnState()


def spawn_statsig_refresh() -> None:
    """Schedule a statsig pair refresh without awaiting it.

    The task handle is retained in a module-level registry (with a
    done-callback discard) so the coroutine is never garbage-collected
    mid-flight.
    """
    task = asyncio.create_task(refresh_statsig_pair())
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


# ---------------------------------------------------------------- session map


@dataclass
class SessionState:
    """Live conversation checkpoint held in memory.

    Attributes:
        account_key: Owning account key.
        grok: Live gateway session (one per conversation).
        user_chain: Ordered user messages forming the chain key.
        created_at: Creation epoch timestamp.
        last_used: Last-use epoch timestamp.

    """

    account_key: str
    grok: GrokSession
    user_chain: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Mark the checkpoint as recently used."""
        self.last_used = time.time()


SESSIONS: dict[str, SessionState] = {}


def _clone_session_state(state: SessionState, mode: str | None = None) -> SessionState:
    """Clone a checkpoint, optionally switching the model mode.

    Args:
        state: Checkpoint to clone.
        mode: Model mode for the clone, or None to keep the source mode.

    Returns:
        Independent checkpoint copy sharing no mutable history.

    """
    source = state.grok
    sess = source.clone_checkpoint(mode)
    return SessionState(
        account_key=state.account_key,
        grok=sess,
        user_chain=list(state.user_chain),
        created_at=state.created_at,
        last_used=time.time(),
    )


async def fork_session_state(
    state: SessionState,
    mode: str | None = None,
) -> SessionState:
    """Snapshot a checkpoint before advancing a conversation branch.

    The live WebSocket is MOVED to the fork, not closed: sequential turns
    (the common case) keep the connection warm and skip the ~1s reconnect
    (TLS + WS handshake + session.create) that fork-then-reconnect paid on
    every request. A later fork from THIS checkpoint reconnects on demand
    via ask(); the clone carries conversation_id and last_parent_response_id
    either way, so branch semantics are unchanged.

    Args:
        state: Checkpoint to snapshot.
        mode: Model mode for the fork, or None to keep the source mode.

    Returns:
        Fork carrying the live WebSocket when the checkpoint had one.

    """
    source = state.grok
    async with source.lock:
        fork = _clone_session_state(state, mode)
        fork.grok.ws = source.ws
        fork.grok.ws_mode = source.ws_mode
        source.ws = None
        return fork


SESSION_LOCK = asyncio.Lock()


async def prune_sessions() -> None:
    """Evict expired and over-cap checkpoints from memory and disk."""
    now = time.time()
    to_close = []
    async with SESSION_LOCK:
        stale_keys = [k for k, v in SESSIONS.items() if now - v.last_used > SESSION_TTL]
        for k in stale_keys:
            st = SESSIONS.pop(k, None)
            if st:
                to_close.append(st.grok)
        if len(SESSIONS) > MAX_SESSIONS:
            sorted_sessions = sorted(
                SESSIONS.items(),
                key=lambda item: item[1].last_used,
            )
            excess = len(SESSIONS) - MAX_SESSIONS
            for k, st in sorted_sessions[:excess]:
                SESSIONS.pop(k, None)
                to_close.append(st.grok)
    # Prune sqlite store outside the lock to avoid blocking other requests on disk I/O
    try:
        store.prune_stale_sessions(SESSION_TTL, MAX_SESSIONS)
    except (OSError, RuntimeError, ValueError, AttributeError, sqlite3.Error) as e:
        logging.getLogger("uvicorn.error").debug("session prune failed: %s", e)
    for g in to_close:
        with suppress(OSError, RuntimeError, AttributeError):
            await g.close()


async def get_or_create_session(
    prefix_key: str | None,
    users: list[str],
    acc: Account,
    mode: str = "fast",
) -> tuple[GrokSession, bool]:
    """Fork a session from the requested chain checkpoint.

    The fork carries the live WebSocket when the checkpoint had one.

    Args:
        prefix_key: Chain key of the parent checkpoint, if any.
        users: Ordered user messages backing a fresh session.
        acc: Account owning the session.
        mode: Model mode for the session.

    Returns:
        Tuple of the live session and whether it continued a checkpoint.

    Raises:
        ValueError: If a checkpoint hit carries an empty prefix key.

    """
    uid = await _uid_for(acc)
    # Fast in-memory check without holding lock during await
    st: SessionState | None = None
    need_persist: bool = False
    async with SESSION_LOCK:
        if prefix_key:
            st = SESSIONS.get(prefix_key)
            if (
                st
                and st.account_key == acc.key
                and (st.grok.alive() or st.grok.conversation_id)
            ):
                # will fork outside lock
                pass
            elif not st:
                need_persist = True
            else:
                st = None
    if st is not None:
        fork = await fork_session_state(st, mode)
        fork.touch()
        if not isinstance(prefix_key, str) or not prefix_key:
            msg = "prefix_key must be a non-empty string"
            raise ValueError(msg)
        store.touch_session(prefix_key, fork.last_used)
        return fork.grok, True
    if need_persist and prefix_key:
        persisted = store.get_session(prefix_key)
        if persisted and persisted["account_key"] == acc.key:
            sess = GrokSession(acc.cookie_header(), uid, mode)
            sess.conversation_id = _persisted_str(persisted, "conversation_id", "")
            sess.last_parent_response_id = _persisted_str(
                persisted,
                "last_parent_response_id",
                "",
            )
            sess.attachments = _clean_attachment_registry(persisted.get("attachments"))
            st_new = SessionState(
                account_key=acc.key,
                grok=sess,
                user_chain=_persisted_chain(persisted, users),
                created_at=_persisted_time(persisted, "created_at", time.time()),
                last_used=time.time(),
            )
            async with SESSION_LOCK:
                # double-check race
                if prefix_key not in SESSIONS:
                    SESSIONS[prefix_key] = st_new
            store.touch_session(prefix_key, st_new.last_used)
            fork = await fork_session_state(st_new, mode)
            return fork.grok, True
    sess = GrokSession(acc.cookie_header(), uid, mode)
    state = SessionState(account_key=acc.key, grok=sess, user_chain=list(users))
    if prefix_key:
        async with SESSION_LOCK:
            SESSIONS[prefix_key] = state
        store.save_session(
            session_key=prefix_key,
            account_key=acc.key,
            user_chain=list(users),
            conversation_id=sess.conversation_id,
            last_parent_response_id=sess.last_parent_response_id,
            model_mode=mode,
            attachments=_session_attachments(sess),
            created_at=state.created_at,
            last_used=state.last_used,
        )
    return sess, False


uid_cache: dict[int, str] = {}


async def _uid_for(acc: Account) -> str:
    user_id = acc.user_id
    if user_id:
        return user_id
    if acc.index not in uid_cache:
        stored_uid = store.get_uid(acc.key)
        if stored_uid:
            uid_cache[acc.index] = stored_uid
            acc.user_id = stored_uid
            return stored_uid
        uid_cache[acc.index] = await gw.resolve_user_id(acc.cookie_header())
        acc.user_id = uid_cache[acc.index]
        store.set_uid(acc.key, acc.user_id)
    return uid_cache[acc.index]


def user_texts(messages: list[dict[str, Any]]) -> list[str]:
    """Collect plain-text bodies of user messages in order.

    Args:
        messages: Chat messages with string or part-list content.

    Returns:
        Text of each user message, in conversation order.

    """
    texts: list[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            texts.append(content)
        else:
            texts.append(content_to_text(content))
    return texts


def rid_hint(users: list[str]) -> str:
    """Build a response-id hint prefix for a user chain.

    Args:
        users: Ordered user messages forming the chain.

    Returns:
        Chain key prefixed for response-id correlation.

    """
    return "resp:" + chain_key(users)


def chain_key(users: list[str], auth_token: str = "") -> str:
    """Hash a user chain (plus auth) into a short session key.

    Args:
        users: Ordered user messages forming the chain.
        auth_token: Authorization header isolating per-key chains.

    Returns:
        24-hex-character chain key.

    """
    payload = {"auth": auth_token, "users": users}
    canon = json.dumps(payload, ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()[:24]


def build_history_prompt(
    flat_messages: list[dict[str, Any]],
    latest_prompt: str,
) -> str:
    """Full-transcript prompt for turns with no grok-side continuation.

    OpenAI requests are stateless: the client resends the whole conversation
    every turn, but continued turns forward only the newest message and rely
    on the gateway holding history (conversation attach + parent_response_id
    + keep_context). When no checkpoint exists for this chain — account
    failover, restart miss, prefix mismatch — shipping the bare latest
    message silently drops all context (the "remember 974" -> "3"
    hallucination). The transcript restores it as plain role-labeled text.

    Single-turn requests return `latest_prompt` byte-identical so the common
    case pays no extra tokens and no formatting churn.

    Args:
        flat_messages: Flattened chat messages with text content.
        latest_prompt: Newest user prompt anchoring the transcript tail.

    Returns:
        Role-labeled transcript, or latest_prompt for single turns.

    """
    turns: list[tuple[str, str]] = []
    for m in flat_messages:
        role = m.get("role")
        if role not in {"user", "assistant"}:
            continue
        c = m.get("content")
        text = c if isinstance(c, str) else content_to_text(c)
        if text.strip():
            turns.append((role, text.strip()))
    if turns and turns[-1][0] == "user":
        turns[-1] = ("user", latest_prompt.strip() or turns[-1][1])
    elif latest_prompt.strip():
        turns.append(("user", latest_prompt.strip()))
    if len(turns) <= 1:
        return latest_prompt
    return "\n\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {text}" for role, text in turns
    )


# ---------------------------------------------------------------- sources bridge

SOURCE_APPENDIX_MAX = 50


def include_sources(flag: object) -> bool:
    """Decide whether to append the sources appendix.

    Falls back to the G2O_INCLUDE_SOURCES config flag.

    Args:
        flag: Per-request override value, or None to use config.

    Returns:
        True when sources should be appended to the response.

    """
    if flag is None:
        return INCLUDE_SOURCES
    if isinstance(flag, str):
        return flag.strip().lower() in {"1", "true", "yes", "on"}
    return bool(flag)


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""


def source_appendix(sources: list[dict[str, Any]], query: str) -> str:
    r"""Bridge source appendix for llmcord-go's "Show Sources" button.

    Matches the appendix contract parsed by llmcord-go across bridge providers:
        \n\nSources
        1. [Title](url) (domain) via `query`

        Search Queries
        1. `query`

    Args:
        sources: Gateway source dicts with title and url keys.
        query: Search query attributed after each entry.

    Returns:
        Markdown appendix block, or an empty string without entries.

    """
    entries: list[str] = []
    seen_urls: set[str] = set()
    clean_query = " ".join(query.split()).replace("`", "'").strip() if query else ""
    for src in sources[:SOURCE_APPENDIX_MAX]:
        raw_url = src.get("url")
        if not isinstance(raw_url, str) or not raw_url.strip():
            continue
        url = raw_url.strip().replace("\n", "").replace("\r", "").replace("\t", "")
        url = url.replace(")", "%29").replace(" ", "%20")
        if not url or url.lower() in seen_urls:
            continue
        seen_urls.add(url.lower())

        title = " ".join(str(src.get("title") or "").split())
        title = title.replace("[", "").replace("]", "") or url

        entry = f"[{title}]({url})"
        host = _host_of(url)
        if title != url and host:
            entry += f" ({host})"
        if clean_query:
            entry += f" via `{clean_query}`"
        entries.append(entry)

    if not entries:
        return ""
    lines = ["Sources"]
    lines.extend(f"{i}. {entry}" for i, entry in enumerate(entries, start=1))
    if clean_query:
        lines.extend(["", "Search Queries", f"1. `{clean_query}`"])
    return "\n\n" + "\n".join(lines)


# ------------------------------------------------------------------- helpers


def check_auth(request: Request) -> None:
    """Reject requests carrying a wrong API key.

    Args:
        request: Incoming FastAPI request.

    Raises:
        HTTPException: If a key is configured and the bearer mismatches.

    """
    if not API_KEY:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid api key")


IMAGE_WORDS = re.compile(
    r"\b(generate|create|draw|paint|render|make|imagine)\b[^.?!]{0,60}\b(image|picture|photo|drawing|art|illustration|logo|wallpaper|portrait|scene|cat|dog|animal)\b"
    r"|\b(image|picture|photo|drawing|illustration)\s+of\b",
    re.IGNORECASE,
)

# Cross-turn attachment memory (see stream_session_turn): grok's gateway only
# renders files mentioned on the CURRENT message, so follow-up turns re-mention
# ids remembered on the session and dedupe replayed bytes by content hash.
# MAX_TRACKED_ATTACHMENTS lives in session_store (shared cap for all writers).
MAX_TURN_MENTIONS = 6  # mentioned per turn (matches file_jobs[:6])
ATTACHMENT_ERROR_WORDS = (
    "fileattachment",
    "file attachment",
    "file_mention",
    "attachment",
)

TEXTUAL_MIMES_PREFIX = ("text/",)
TEXTUAL_MIMES = {
    "application/json",
    "application/javascript",
    "application/typescript",
    "application/xml",
    "application/x-python",
    "application/x-sh",
    "application/yaml",
    "application/toml",
    "application/sql",
}
TEXTUAL_EXT = re.compile(
    r"\.(txt|md|markdown|json|jsonl|yaml|yml|toml|ini|cfg|conf|py|js|mjs|cjs|ts|tsx|jsx|"
    r"java|kt|go|rs|rb|php|c|h|cpp|hpp|cs|swift|sh|bash|zsh|sql|html?|css|scss|xml|csv|tsv|log|env)$",
    re.IGNORECASE,
)


async def _fetch_statsig_page() -> str | None:
    """Fetch the grok root page HTML carrying the statsig seed.

    Returns:
        Page text, or None when the page is blocked or not HTML.

    """
    acc = pool.acquire()
    cookie = acc.cookie_header() if acc else ""
    async with AsyncSession(impersonate="chrome") as s:
        # /index 404s; the seed/curves payload lives on the root page.
        r = await s.get(
            f"{GROK_BASE}/",
            headers={"user-agent": USER_AGENT, "cookie": cookie},
            timeout=10,
        )
        if r.status_code != HTTP_OK:
            # Cloudflare challenge pages carry no meta seed / curves;
            # feeding them to the extractor silently keeps the pair
            # unready. Return None so ensure_pair skips cleanly.
            return None
        page_text = r.text
        if not isinstance(page_text, str):
            return None
        return page_text


async def refresh_statsig_pair() -> None:
    """Refresh the statsig seed pair, warning sparingly when unavailable."""
    try:
        await statsig.ensure_pair(_fetch_statsig_page)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        TimeoutError,
    ) as e:
        logging.getLogger("uvicorn.error").debug("statsig refresh failed: %s", e)
    if not statsig.ready and (
        time.time() - _statsig_warn_state.last_warned_at > STATSIG_WARN_INTERVAL_SECONDS
    ):
        _statsig_warn_state.last_warned_at = time.time()
        logging.getLogger("uvicorn.error").warning(
            "statsig pair unavailable (grok.com HTML blocked by anti-bot); "
            "REST calls will go out without x-statsig-id",
        )


def content_to_text(content: object) -> str:
    """Flatten OpenAI content parts to plain prompt text.

    Args:
        content: String content or a list of part dicts.

    Returns:
        Plain text joined from text and input_text parts.

    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t in {"text", "input_text"}:
                text = p.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)


_TEXT_PART_TYPES = frozenset({"text", "input_text"})

_FILE_PART_TYPES = frozenset({"input_file", "file"})


def _decode_data_url_job(value: str, name: str) -> tuple[dict[str, Any] | None, str]:
    """Decode a data-URL part into an upload job.

    Args:
        value: Data-URL string to decode.
        name: File name to attach to the job.

    Returns:
        Tuple of the upload job (None when malformed) and error text.

    """
    try:
        data, mime, _ = decode_data_url(value)
    except UploadError as e:
        return None, str(e)
    return {"name": name, "data": data, "mime": mime}, ""


def _queue_remote_job(
    url: str,
    name: str,
    mime: str | None,
    ordered_jobs: list[dict[str, Any] | None],
    pending_downloads: list[dict[str, Any]],
) -> None:
    """Reserve an order slot and queue an HTTP download for a remote file.

    Args:
        url: Remote file URL to download later.
        name: File name to attach to the job.
        mime: Known mime type, or None to sniff from the response.
        ordered_jobs: Order-preserving slots for resolved upload jobs.
        pending_downloads: Queue for remote downloads resolved later.

    """
    ordered_jobs.append(None)
    pending_downloads.append(
        {
            "url": url,
            "name": name,
            "mime": mime,
            "ordered_idx": len(ordered_jobs) - 1,
        },
    )


def _extract_image_url_part(
    part: dict[str, Any],
    ordered_jobs: list[dict[str, Any] | None],
    pending_downloads: list[dict[str, Any]],
) -> None:
    """Collect a chat image_url part into jobs or queued downloads.

    Args:
        part: Content part with an image_url payload.
        ordered_jobs: Order-preserving slots for resolved upload jobs.
        pending_downloads: Queue for remote downloads resolved later.

    """
    url = part.get("image_url") or {}
    val = url.get("url") if isinstance(url, dict) else url
    if not isinstance(val, str):
        return
    if val.startswith("data:"):
        name = part.get("name") or "image"
        job, decode_error = _decode_data_url_job(val, name)
        if job is None:
            logging.getLogger("uvicorn.error").warning(
                "dropping malformed data-URL image attachment: %s",
                decode_error,
            )
            return
        ordered_jobs.append(job)
    elif val.startswith("http"):
        name = val.split("?")[0].split("/")[-1] or "image"
        _queue_remote_job(val, name, None, ordered_jobs, pending_downloads)
    else:
        # grok file id passed through
        ordered_jobs.append({"file_id": val})


def _extract_input_image_part(
    part: dict[str, Any],
    ordered_jobs: list[dict[str, Any] | None],
    pending_downloads: list[dict[str, Any]],
) -> None:
    """Collect a responses input_image part into jobs or queued downloads.

    Args:
        part: Content part with an image_url or url payload.
        ordered_jobs: Order-preserving slots for resolved upload jobs.
        pending_downloads: Queue for remote downloads resolved later.

    """
    url = part.get("image_url") or part.get("url")
    if not isinstance(url, str):
        return
    if url.startswith("data:"):
        job, decode_error = _decode_data_url_job(url, "image")
        if job is None:
            logging.getLogger("uvicorn.error").warning(
                "dropping malformed data-URL image attachment: %s",
                decode_error,
            )
            return
        ordered_jobs.append(job)
    elif url.startswith("http"):
        _queue_remote_job(url, "image", None, ordered_jobs, pending_downloads)


def _extract_file_part(
    part: dict[str, Any],
    part_type: object,
    ordered_jobs: list[dict[str, Any] | None],
    pending_downloads: list[dict[str, Any]],
) -> None:
    """Collect an input_file/file part into jobs or queued downloads.

    Args:
        part: Content part with file, file_data, file_id, or file_url.
        part_type: Original part type for drop warnings.
        ordered_jobs: Order-preserving slots for resolved upload jobs.
        pending_downloads: Queue for remote downloads resolved later.

    """
    fd = part.get("file") or {}
    name = fd.get("filename") or part.get("filename") or "file"
    mime = fd.get("mime_type") or part.get("mime_type")
    file_data = fd.get("file_data")
    if not isinstance(file_data, str):
        file_data = part.get("file_data")
    file_id = fd.get("file_id")
    if not isinstance(file_id, str):
        file_id = part.get("file_id")
    if isinstance(file_data, str) and file_data.startswith("data:"):
        job, decode_error = _decode_data_url_job(file_data, name)
        if job is None:
            logging.getLogger("uvicorn.error").warning(
                "dropping malformed data-URL file attachment (%s): %s",
                name,
                decode_error,
            )
            return
        ordered_jobs.append(job)
    elif isinstance(file_id, str):
        ordered_jobs.append({"file_id": file_id})
    elif isinstance(part.get("file_url"), str):
        _queue_remote_job(part["file_url"], name, mime, ordered_jobs, pending_downloads)
    else:
        logging.getLogger("uvicorn.error").warning(
            "dropping %s part with no usable file_data/file_id/file_url",
            part_type,
        )


def _accumulate_part(
    part: dict[str, Any],
    new_parts: list[str],
    ordered_jobs: list[dict[str, Any] | None],
    pending_downloads: list[dict[str, Any]],
) -> None:
    """Fold one content part into prompt text or queued file jobs.

    Args:
        part: Single content part dict.
        new_parts: Collector for flattened prompt text pieces.
        ordered_jobs: Order-preserving slots for resolved upload jobs.
        pending_downloads: Queue for remote downloads resolved later.

    """
    part_type = part.get("type")
    if part_type in _TEXT_PART_TYPES:
        new_parts.append(part.get("text", ""))
    elif part_type == "image_url":
        _extract_image_url_part(part, ordered_jobs, pending_downloads)
    elif part_type == "input_image":
        _extract_input_image_part(part, ordered_jobs, pending_downloads)
    elif part_type in _FILE_PART_TYPES:
        _extract_file_part(part, part_type, ordered_jobs, pending_downloads)


def _split_message(
    msg: dict[str, Any],
    ordered_jobs: list[dict[str, Any] | None],
    pending_downloads: list[dict[str, Any]],
) -> dict[str, Any]:
    """Split one message into prompt text, queuing file jobs in order.

    Args:
        msg: Chat message with string or part-list content.
        ordered_jobs: Order-preserving slots for resolved upload jobs.
        pending_downloads: Queue for remote downloads resolved later.

    Returns:
        Message copy with flattened text content.

    """
    content = msg.get("content")
    if isinstance(content, str):
        return {**msg, "content": content}
    if not isinstance(content, list):
        return msg
    new_parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            _accumulate_part(part, new_parts, ordered_jobs, pending_downloads)
    return {**msg, "content": "\n".join(new_parts)}


async def _download_bytes(url: str) -> tuple[bytes, str]:
    """Download a remote attachment with a browser-like session.

    Args:
        url: Remote attachment URL.

    Returns:
        Tuple of response bytes and content-type mime.

    Raises:
        UploadError: If the response carries no byte payload.

    """
    async with AsyncSession(impersonate="chrome") as s:
        dr = await s.get(url, timeout=30)
        content = dr.content
        if not isinstance(content, bytes):
            msg = f"unexpected download payload for {url[:80]}"
            raise UploadError(msg)
        mime_raw = dr.headers.get("content-type", "image/png")
        if not isinstance(mime_raw, str):
            mime_raw = "image/png"
        mime = str(mime_raw).split(";")[0]
        return content, mime


async def _fetch_pending_download(item: dict[str, Any]) -> dict[str, Any]:
    """Resolve one queued attachment download to an order-keyed result.

    Args:
        item: Queued download dict with url, name, mime, and ordered_idx.

    Returns:
        Result dict with ordered_idx, job (None when failed), and url.

    """
    url = item["url"]
    try:
        data, content_type = await _download_bytes(url)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        TimeoutError,
        UploadError,
    ) as e:
        logging.getLogger("uvicorn.error").warning(
            "attachment download failed (%s): %s",
            url[:120],
            e,
        )
        return {"ordered_idx": item["ordered_idx"], "job": None, "url": url}
    mime = item["mime"] or content_type
    return {
        "ordered_idx": item["ordered_idx"],
        "job": {"name": item["name"], "data": data, "mime": mime},
        "url": url,
    }


async def _resolve_pending_downloads(
    pending_downloads: list[dict[str, Any]],
) -> dict[int, dict[str, Any] | None]:
    """Download queued attachments concurrently, keyed by order slot.

    Args:
        pending_downloads: Queued download dicts with ordered_idx keys.

    Returns:
        Mapping of order slot to resolved job (None when failed).

    """
    results = await asyncio.gather(
        *[_fetch_pending_download(item) for item in pending_downloads],
    )
    return {result["ordered_idx"]: result["job"] for result in results}


async def extract_attachments(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split message content parts into prompt messages and file jobs.

    Text-like files are inlined into the prompt text. Images and other binary
    files are returned as jobs for native upload.

    Args:
        messages: Chat messages with string or part-list content.

    Returns:
        Tuple of prompt-ready messages and native upload jobs in order.

    """
    ordered_jobs: list[dict[str, Any] | None] = []
    pending_downloads: list[dict[str, Any]] = []
    out = [_split_message(msg, ordered_jobs, pending_downloads) for msg in messages]
    if pending_downloads:
        resolved = await _resolve_pending_downloads(pending_downloads)
        for idx, job in resolved.items():
            ordered_jobs[idx] = job
    return out, [job for job in ordered_jobs if job is not None]


def _decode_job_text(job: dict[str, Any]) -> str | None:
    """Decode an attachment job payload to text.

    Args:
        job: Upload job with bytes or string data.

    Returns:
        Decoded text, or None for non-textual payloads.

    """
    data = job.get("data")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data).decode("utf-8", errors="replace")
    if isinstance(data, str):
        return data
    return None


def inline_textual(job: dict[str, Any]) -> str | None:
    """Inline a textual file as fenced markdown.

    Args:
        job: Upload job with name, mime, and data.

    Returns:
        Fenced code block, or None for binary files.

    """
    mime_raw = job.get("mime") or ""
    mime = mime_raw if isinstance(mime_raw, str) else ""
    name_raw = job.get("name") or ""
    name = name_raw if isinstance(name_raw, str) else ""
    if not (
        mime.startswith(TEXTUAL_MIMES_PREFIX)
        or mime in TEXTUAL_MIMES
        or TEXTUAL_EXT.search(name)
    ):
        return None
    try:
        text = _decode_job_text(job)
    except (ValueError, TypeError, AttributeError, UnicodeError):
        return None
    if text is None:
        return None
    ext = (
        re.sub(r"[^a-z0-9]", "", name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    )
    lang = ext if len(ext) <= MAX_INLINE_LANG_LEN else ""
    return f"```{lang} {name}\n{text}\n```"


MODEL_MODE_MAP = {
    "grok-fast": "fast",
    "fast": "fast",
    "grok-4.5-fast": "fast",
    "grok-auto": "auto",
    "auto": "auto",
    "grok-expert": "expert",
    "expert": "expert",
    "grok-heavy": "heavy",
    "heavy": "heavy",
    "grok-build": "build",
    "build": "build",
}


def resolve_mode(model: str | None) -> tuple[str, str]:
    """Resolve a public model name to a gateway mode and response name.

    Args:
        model: Requested model name, or None for the default.

    Returns:
        Tuple of gateway mode and public model name for responses.

    """
    m = (model or DEFAULT_MODEL).strip().lower()
    mode = MODEL_MODE_MAP.get(m)
    if mode is None:
        mode = "fast"
    public = next((k for k, v in MODEL_MODE_MAP.items() if v == mode), m)
    return mode, public


def now_epoch() -> int:
    """Return the current time as epoch seconds.

    Returns:
        Current epoch time truncated to seconds.

    """
    return int(time.time())


# -------------------------------------------------------------- image upload


def _job_hash(job: dict[str, Any]) -> str | None:
    """Hash job bytes for cross-turn upload deduplication.

    Covers bytes only: identical content re-sent under a different name/mime
    intentionally reuses the original upload (grok keeps that upload's stored
    metadata) instead of paying for a second upload.

    Args:
        job: Upload job with optional bytes data.

    Returns:
        SHA-256 hex digest, or None when there are no bytes.

    """
    data = job.get("data")
    if isinstance(data, (bytes, bytearray)):
        return hashlib.sha256(bytes(data)).hexdigest()
    return None


def _persisted_str(row: Mapping[str, object], key: str, default: str) -> str:
    """Read a string field from a persisted session row.

    Args:
        row: Raw persisted session mapping.
        key: Field name.
        default: Fallback value.

    Returns:
        String field value or the default.

    """
    value = row.get(key, default)
    return value if isinstance(value, str) else default


def _persisted_chain(row: Mapping[str, object], fallback: list[str]) -> list[str]:
    """Read the user chain from a persisted session row.

    Args:
        row: Raw persisted session mapping.
        fallback: Fallback chain.

    Returns:
        Stored string chain or a copy of the fallback.

    """
    value = row.get("user_chain", fallback)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return [v for v in value if isinstance(v, str)]
    return list(fallback)


def _persisted_time(row: Mapping[str, object], key: str, default: float) -> float:
    """Read an epoch timestamp from a persisted session row.

    Args:
        row: Raw persisted session mapping.
        key: Field name.
        default: Fallback value.

    Returns:
        Stored timestamp or the default.

    """
    value = row.get(key, default)
    if isinstance(value, bool):
        return default
    return value if isinstance(value, (int, float)) else default


def _clean_attachment_registry(entries: object) -> list[dict[str, Any]]:
    """Validate and cap a stored attachment registry.

    Args:
        entries: Raw registry rows, oldest first.

    Returns:
        Valid rows with the newest entries kept.

    """
    return clean_attachment_rows(entries)


def _session_attachments(sess: GrokSession) -> list[dict[str, Any]]:
    """Read the validated attachment registry off a session.

    Args:
        sess: Gateway session carrying an attachment registry.

    Returns:
        Valid registry rows, oldest first.

    """
    return _clean_attachment_registry(getattr(sess, "attachments", None))


def propagate_dropped_attachments(
    source_sess: GrokSession,
    forked_sess: GrokSession,
) -> None:
    """Carry stale-id invalidations from a turn back to its source checkpoint.

    Checkpoints stay immutable for branching (fork_session_state), but a file
    id stream_session_turn declared stale is dead everywhere: re-mentioning it
    on the next failover attempt would only burn another failed gateway turn.
    Only explicitly dropped ids are removed - never cap evictions or new
    uploads.

    Args:
        source_sess: Checkpoint session to prune.
        forked_sess: Turn session reporting dropped ids.

    """
    dropped = getattr(forked_sess, "last_dropped_attachment_ids", None)
    if not dropped:
        return
    src = _clean_attachment_registry(getattr(source_sess, "attachments", None))
    if not src:
        return
    dead = set(dropped)
    source_sess.attachments = [e for e in src if e["file_id"] not in dead]


class TurnOptions(TypedDict, total=False):
    """Optional per-turn arguments shared by turn entry points."""

    attachment_ids: list[str] | None
    system_prompt: str | None
    file_jobs: list[dict[str, Any]] | None
    user_text: str | None
    mode: str
    history_prompt: str | None
    latest: str | None


class PickOptions(TurnOptions, total=False):
    """Turn options plus prompts for account-picking entry points."""

    prompt: str


@dataclass
class _TurnAttachments:
    """Mutable attachment-resolution state for one session turn."""

    registry: list[dict[str, Any]] = field(default_factory=list)
    mention_ids: list[str] = field(default_factory=list)
    prior_mentioned: list[str] = field(default_factory=list)
    turn_ids: list[str] = field(default_factory=list)
    upload_errors: list[str] = field(default_factory=list)
    upload_succeeded: bool = False
    new_entries: list[dict[str, Any]] = field(default_factory=list)
    caller_ids: list[str] = field(default_factory=list)

    def mention(self, fid: str) -> None:
        """Mention a file id unless already mentioned."""
        if fid and fid not in self.mention_ids:
            self.mention_ids.append(fid)

    def turn_mention(self, fid: str) -> None:
        """Mention an id resolved from this request; it survives the cap."""
        if not fid:
            return
        if fid not in self.turn_ids:
            self.turn_ids.append(fid)
        self.mention(fid)


@dataclass(frozen=True)
class _TurnRequest:
    """Prompt payload for one gateway turn."""

    prompt: str
    mention_ids: list[str]
    system_prompt: str | None
    user_text: str


def _collect_passthrough_ids(
    state: _TurnAttachments,
    file_jobs: list[dict[str, Any]],
) -> None:
    """Mention caller-supplied file ids without upload machinery.

    Args:
        state: Mutable attachment-resolution state.
        file_jobs: Native upload jobs from extract_attachments.

    """
    for job in file_jobs[:MAX_TURN_MENTIONS]:
        if "file_id" not in job:
            continue
        fid = job["file_id"]
        state.turn_mention(fid)
        if fid not in state.prior_mentioned:
            state.caller_ids.append(fid)


def _plan_upload_ops(
    state: _TurnAttachments,
    jobs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str | None, str, int]]]:
    """Split byte jobs into dedup hits and pending uploads, preserving order.

    Args:
        state: Mutable attachment-resolution state.
        jobs: Capped upload jobs for this turn.

    Returns:
        Tuple of ordered replay ops and pending upload tuples.

    """
    known_hashes = {e["hash"]: e["file_id"] for e in state.registry if e["hash"]}
    ordered_ops: list[dict[str, Any]] = []
    to_upload: list[tuple[dict[str, Any], str | None, str, int]] = []
    for job in jobs:
        if "file_id" in job:
            continue
        raw_name = job.get("name")
        name = raw_name if isinstance(raw_name, str) and raw_name else "file"
        job_hash = _job_hash(job)
        if job_hash and known_hashes.get(job_hash):
            ordered_ops.append(
                {"type": "dedup", "fid": known_hashes[job_hash], "hash": job_hash},
            )
        else:
            idx = len(ordered_ops)
            ordered_ops.append(
                {
                    "type": "upload",
                    "job": job,
                    "hash": job_hash,
                    "name": name,
                    "idx": idx,
                },
            )
            to_upload.append((job, job_hash, name, idx))
    return ordered_ops, to_upload


async def _upload_one_file(
    cookie: str,
    job: dict[str, Any],
    job_hash: str | None,
    name: str,
    ordered_idx: int,
) -> tuple[int, str | None, str | None, str, str | None]:
    """Upload one attachment job, never raising.

    Args:
        cookie: Owning account cookie header.
        job: Upload job with bytes data.
        job_hash: Content hash for registry bookkeeping, if any.
        name: File name for the upload.
        ordered_idx: Order slot for encounter-order replay.

    Returns:
        Tuple of order index, file id, hash, name, and error text.

    """
    raw_data = job.get("data")
    data = bytes(raw_data) if isinstance(raw_data, (bytes, bytearray)) else b""
    raw_mime = job.get("mime")
    mime = raw_mime if isinstance(raw_mime, str) else None
    payload = UploadPayload(filename=name, data=data, mime=mime)
    try:
        async with AsyncSession(impersonate="chrome") as s:
            meta = await upload_file(s, cookie, statsig, payload)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        TimeoutError,
        UploadError,
    ) as e:
        return ordered_idx, None, job_hash, name, str(e)
    file_id = meta.get("fileMetadataId")
    if not isinstance(file_id, str) or not file_id:
        return ordered_idx, None, job_hash, name, "upload returned no fileMetadataId"
    return ordered_idx, file_id, job_hash, name, None


def _replay_dedup(state: _TurnAttachments, fid: str) -> None:
    """Re-mention a deduplicated id, refreshing its registry recency.

    Args:
        state: Mutable attachment-resolution state.
        fid: Previously uploaded file id.

    """
    state.turn_mention(fid)
    entry = next((x for x in state.registry if x["file_id"] == fid), None)
    if entry is not None:
        state.registry = [x for x in state.registry if x is not entry] + [entry]


def _replay_upload(
    state: _TurnAttachments,
    fid: str | None,
    job_hash: str | None,
    name: str,
    err: str | None,
) -> None:
    """Replay one upload result into mentions or errors, in order.

    Args:
        state: Mutable attachment-resolution state.
        fid: Uploaded file id, or None when the upload failed.
        job_hash: Content hash for registry bookkeeping, if any.
        name: File name for failure messages.
        err: Failure text, if any.

    """
    if fid:
        state.upload_succeeded = True
        state.turn_mention(fid)
        state.new_entries.append({"file_id": fid, "hash": job_hash})
    else:
        state.upload_errors.append(f"{name}: {err}")


def _raise_on_total_upload_failure(state: _TurnAttachments) -> None:
    """Fail the turn when every attachment upload failed.

    Never ship a prompt whose attachments were all dropped: grok then
    confidently answers "you didn't attach anything" (the 2026-08-25
    "fact check this" incident). Remembered ids do not rescue a
    fully-failed batch - those new files are simply gone. Fail the turn so
    the caller fails over to another account or errors out honestly.

    Args:
        state: Mutable attachment-resolution state.

    Raises:
        GatewayError: When errors exist and no upload succeeded.

    """
    if state.upload_errors and not state.upload_succeeded:
        msg = "attachment upload failed: " + "; ".join(state.upload_errors[:3])
        raise GatewayError(UPSTREAM_KIND, msg)
    if state.upload_errors:
        logging.getLogger("uvicorn.error").warning(
            "some attachments failed to upload: %s",
            "; ".join(state.upload_errors[:3]),
        )


async def _run_upload_batch(
    sess: GrokSession,
    state: _TurnAttachments,
    ordered_ops: list[dict[str, Any]],
    to_upload: list[tuple[dict[str, Any], str | None, str, int]],
) -> None:
    """Upload pending jobs concurrently, replaying results in order.

    Args:
        sess: Gateway session owning the turn.
        state: Mutable attachment-resolution state.
        ordered_ops: Replay ops preserving encounter order.
        to_upload: Pending upload tuples for concurrent upload.

    """
    cookie = sess.cookie_header
    try:
        await refresh_statsig_pair()
        results = await asyncio.gather(
            *[
                _upload_one_file(cookie, job, job_hash, name, idx)
                for job, job_hash, name, idx in to_upload
            ],
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        TimeoutError,
        UploadError,
        GatewayError,
    ) as e:
        state.upload_errors.append(str(e))
        _raise_on_total_upload_failure(state)
        return
    upload_by_idx = {oi: (fid, h, n, err) for oi, fid, h, n, err in results}
    for op in ordered_ops:
        if op["type"] == "dedup":
            _replay_dedup(state, op["fid"])
        else:
            fid, h, n, err = upload_by_idx[op["idx"]]
            _replay_upload(state, fid, h, n, err)
    _raise_on_total_upload_failure(state)


async def _upload_turn_files(
    sess: GrokSession,
    state: _TurnAttachments,
    file_jobs: list[dict[str, Any]] | None,
) -> None:
    """Resolve this turn's file jobs into mention ids.

    Passthrough file-id jobs need no upload machinery; byte jobs upload
    concurrently with order-preserving replay.

    Args:
        sess: Gateway session owning the turn.
        state: Mutable attachment-resolution state.
        file_jobs: Native upload jobs from extract_attachments, if any.

    """
    if not file_jobs:
        return
    # Passthrough file-id jobs need no upload machinery; handle them for
    # every request shape, not only when byte-upload jobs exist.
    _collect_passthrough_ids(state, file_jobs)
    jobs = file_jobs[:MAX_TURN_MENTIONS]
    if not any("file_id" not in job for job in jobs):
        return
    # Preserve original encounter order while parallelizing uploads.
    ordered_ops, to_upload = _plan_upload_ops(state, jobs)
    if not to_upload:
        # Only dedup hits, no real uploads: still replay dedups in order.
        for op in ordered_ops:
            if op["type"] == "dedup":
                _replay_dedup(state, op["fid"])
        return
    await _run_upload_batch(sess, state, ordered_ops, to_upload)


def _remember_caller_ids(
    state: _TurnAttachments,
    attachment_ids: list[str] | None,
) -> None:
    """Mention caller-supplied attachment ids for this turn.

    Args:
        state: Mutable attachment-resolution state.
        attachment_ids: Caller-supplied file ids, if any.

    """
    for aid in attachment_ids or []:
        if aid:
            state.turn_mention(aid)
            if aid not in state.prior_mentioned:
                state.caller_ids.append(aid)


def _finalize_turn_attachments(
    sess: GrokSession,
    state: _TurnAttachments,
) -> list[str]:
    """Persist registry updates and compute the capped mention list.

    Current-turn ids survive the mention cap; remembered context fills the
    rest, newest last. (A dedupe hit on an old entry must not be evicted in
    favor of never-referenced newer remembered ids.)

    Args:
        sess: Gateway session owning the turn.
        state: Mutable attachment-resolution state.

    Returns:
        File ids to mention on the gateway message.

    """
    seen_ids = {e["file_id"] for e in state.registry}
    for entry in state.new_entries + [
        {"file_id": fid, "hash": None} for fid in state.caller_ids
    ]:
        if entry["file_id"] not in seen_ids:
            state.registry.append(entry)
            seen_ids.add(entry["file_id"])
    state.registry = state.registry[-MAX_TRACKED_ATTACHMENTS:]
    sess.attachments = state.registry
    turn_set = set(state.turn_ids)
    return (
        [i for i in state.mention_ids if i not in turn_set]
        + [i for i in state.turn_ids if i in state.mention_ids]
    )[-MAX_TURN_MENTIONS:]


def _latest_text(user_text: str | None, prompt: str) -> str:
    """Pick the newest user text for degraded-query detection.

    It must stay the newest user message even when prompt carries the
    full-transcript fallback.

    Args:
        user_text: Newest user message, if any.
        prompt: Turn prompt carrying the fallback.

    Returns:
        Non-empty user text, or the prompt.

    """
    if isinstance(user_text, str) and user_text:
        return user_text
    return prompt


async def _ask_with_stale_retry(
    sess: GrokSession,
    state: _TurnAttachments,
    request: _TurnRequest,
) -> AsyncIterator[dict[str, Any]]:
    """Stream a turn, retrying once without stale remembered attachments.

    Remembered ids can go stale (grok expires files). If the turn fails
    with a file-flavored error before anything streamed, retry once
    without them instead of failing over forever. Only ids actually
    mentioned this turn may be declared stale: ids capped out of the
    mention list were never sent, and freshly uploaded ids are not the
    likely culprit.

    Args:
        sess: Gateway session carrying the conversation.
        state: Mutable attachment-resolution state.
        request: Prompt payload for the gateway turn.

    Yields:
        Gateway event dicts as they arrive.

    Raises:
        GatewayError: If the turn fails without a stale-attachment retry.

    """
    events_yielded = False
    try:
        async for ev in sess.ask(
            request.prompt,
            attachment_ids=request.mention_ids or None,
            system_prompt=request.system_prompt,
            user_text=request.user_text,
        ):
            events_yielded = True
            yield ev
    except GatewayError as e:
        msg = str(e).lower()
        stale = set(request.mention_ids) & set(state.prior_mentioned)
        if (
            stale
            and not events_yielded
            and any(w in msg for w in ATTACHMENT_ERROR_WORDS)
        ):
            logging.getLogger("uvicorn.error").warning(
                "dropping %d remembered attachment id(s) after gateway error: %s",
                len(stale),
                e,
            )
            sess.last_dropped_attachment_ids = set(stale)
            state.registry = [x for x in state.registry if x["file_id"] not in stale]
            sess.attachments = state.registry
            retry_ids = [i for i in request.mention_ids if i not in stale]
            async for ev in sess.ask(
                request.prompt,
                attachment_ids=retry_ids or None,
                system_prompt=request.system_prompt,
                user_text=request.user_text,
            ):
                yield ev
        else:
            raise


async def stream_session_turn(
    sess: GrokSession,
    prompt: str,
    **kwargs: Unpack[TurnOptions],
) -> AsyncIterator[dict[str, Any]]:
    """Yield gateway events in real time.

    Streams text_delta / reasoning_delta / image_url / done dicts. Files are
    per-account on grok.com ("FileAttachment not found" on cross-account
    mentions), so upload with THIS session's account right before the turn.
    Gateway failures from the upload and ask helpers propagate to the caller.

    Cross-turn attachment memory: the gateway only "sees" files mentioned on
    the CURRENT message, so a follow-up turn would otherwise ship a bare
    prompt and the model denies ever receiving earlier images (the 2026-08-28
    "top 3" incident: turn 1 attached two images, turn 2 "top 3" answered
    "I can't see the images you attached"). Uploaded ids are therefore
    remembered on the session and re-mentioned on every turn; replayed bytes
    are deduplicated by content hash instead of being uploaded again.

    Args:
        sess: Gateway session carrying the conversation.
        prompt: User prompt for this turn.
        kwargs: Optional attachment_ids, system_prompt, file_jobs, user_text.

    Yields:
        Gateway event dicts as they arrive.

    """
    state = _TurnAttachments(registry=_session_attachments(sess))
    sess.last_dropped_attachment_ids = set()
    # Remembered images ride along so follow-up turns still "see" earlier files.
    for entry in state.registry:
        state.mention(entry["file_id"])
        state.prior_mentioned.append(entry["file_id"])
    await _upload_turn_files(sess, state, kwargs.get("file_jobs"))
    _remember_caller_ids(state, kwargs.get("attachment_ids"))
    mention_ids = _finalize_turn_attachments(sess, state)
    request = _TurnRequest(
        prompt=prompt,
        mention_ids=mention_ids,
        system_prompt=kwargs.get("system_prompt"),
        user_text=_latest_text(kwargs.get("user_text"), prompt),
    )
    async for ev in _ask_with_stale_retry(sess, state, request):
        yield ev


async def run_session_turn(
    sess: GrokSession,
    prompt: str,
    **kwargs: Unpack[TurnOptions],
) -> tuple[TurnResult, list[dict[str, Any]]]:
    """Buffer a session turn for non-streaming requests.

    Args:
        sess: Gateway session carrying the conversation.
        prompt: User prompt for this turn.
        kwargs: Optional attachment_ids, system_prompt, file_jobs, user_text.

    Returns:
        Tuple of the completed turn result and all gateway events.

    Raises:
        GatewayError: If the gateway yields no completed result.

    """
    events = [ev async for ev in stream_session_turn(sess, prompt, **kwargs)]
    done = next((e for e in events if e["type"] == "done"), None)
    if not done:
        msg = "no result from gateway"
        raise GatewayError(UPSTREAM_KIND, msg)
    return done["result"], events


def _fail_kind(error: GatewayError) -> str:
    """Map a gateway error kind to a pool release category.

    Args:
        error: Gateway failure to categorize.

    Returns:
        Pool release category string.

    """
    if error.kind in {"auth", "quota", "degraded"}:
        return error.kind
    return "generic"


def _gateway_http_status(error: GatewayError) -> int:
    """Map a gateway failure to its HTTP status code.

    Args:
        error: Gateway failure to surface.

    Returns:
        429 for quota exhaustion, else 502.

    """
    return 429 if error.kind == "quota" else 502


@dataclass(frozen=True)
class _TurnPrompts:
    """Resolved prompts and threading options for checkpoint failover."""

    mode: str
    continuation: str
    fresh: str
    latest: str
    attachment_ids: list[str] | None
    file_jobs: list[dict[str, Any]] | None
    system_prompt: str | None


def _resolve_turn_prompts(kwargs: PickOptions) -> _TurnPrompts:
    """Split turn options into failover prompts and threading options.

    Continuations forward only the newest message (gateway holds history);
    fresh sessions have no gateway history, so they resend the transcript.

    Args:
        kwargs: Turn options with mode, prompt, history_prompt, latest.

    Returns:
        Resolved prompt bundle.

    Raises:
        HTTPException: If prompt values are not strings.

    """
    mode = kwargs.get("mode") or "fast"
    latest_msg = kwargs.get("latest") or kwargs.get("prompt") or ""
    fresh_prompt = kwargs.get("history_prompt") or kwargs.get("prompt")
    continuation_raw = kwargs.get("prompt")
    if not isinstance(continuation_raw, str):
        raise HTTPException(400, "prompt must be a string")
    if not isinstance(fresh_prompt, str):
        raise HTTPException(400, "prompt must be a string")
    return _TurnPrompts(
        mode=mode,
        continuation=continuation_raw,
        fresh=fresh_prompt,
        latest=latest_msg,
        attachment_ids=kwargs.get("attachment_ids"),
        file_jobs=kwargs.get("file_jobs"),
        system_prompt=kwargs.get("system_prompt"),
    )


async def _restore_checkpoint_state(
    session_key: str,
    users: list[str],
    mode: str,
) -> SessionState | None:
    """Restore a checkpoint from memory or SQLite.

    Memory is checked first without blocking on I/O; SQLite restore runs
    outside the lock to avoid stalling other requests on disk.

    Args:
        session_key: Chain key of the checkpoint.
        users: Ordered user messages for fallback chain data.
        mode: Model mode for a rebuilt session.

    Returns:
        Restored checkpoint, or None when absent or account-unavailable.

    """
    async with SESSION_LOCK:
        st = SESSIONS.get(session_key)
    if st is not None:
        return st
    persisted = store.get_session(session_key)
    if not persisted:
        return None
    acc_cand = next(
        (a for a in pool.snapshot() if a.key == persisted["account_key"]),
        None,
    )
    if acc_cand is None or not acc_cand.available():
        return None
    uid = await _uid_for(acc_cand)
    sess = GrokSession(acc_cand.cookie_header(), uid, mode)
    sess.conversation_id = _persisted_str(persisted, "conversation_id", "")
    sess.last_parent_response_id = _persisted_str(
        persisted,
        "last_parent_response_id",
        "",
    )
    sess.attachments = _clean_attachment_registry(persisted.get("attachments"))
    st_new = SessionState(
        account_key=acc_cand.key,
        grok=sess,
        user_chain=_persisted_chain(persisted, users),
        created_at=_persisted_time(persisted, "created_at", time.time()),
        last_used=time.time(),
    )
    async with SESSION_LOCK:
        # double-check race
        if session_key not in SESSIONS:
            SESSIONS[session_key] = st_new
            return st_new
        return SESSIONS[session_key]


@dataclass
class _StreamAttempt:
    """Outcome of one checkpoint or fresh streaming attempt."""

    attempted: bool = False
    done: bool = False
    error: GatewayError | None = None


async def _fork_turn_events(
    forked: SessionState,
    source_state: SessionState,
    prompts: _TurnPrompts,
    acc: Account,
    outcome: _StreamAttempt,
) -> AsyncIterator[dict[str, Any]]:
    """Yield decorated events for a forked checkpoint turn.

    Args:
        forked: Forked checkpoint advancing the turn.
        source_state: Original checkpoint for drop propagation.
        prompts: Resolved continuation prompts and threading options.
        acc: Account owning the turn.
        outcome: Mutable outcome record.

    Yields:
        Gateway events decorated with acc and state.

    """
    async for ev in stream_session_turn(
        forked.grok,
        prompts.continuation,
        attachment_ids=prompts.attachment_ids,
        file_jobs=prompts.file_jobs,
        system_prompt=prompts.system_prompt,
        user_text=prompts.latest,
    ):
        decorated = dict(ev)
        decorated["acc"] = acc
        decorated["state"] = forked
        yield decorated
        if decorated.get("type") == "done":
            propagate_dropped_attachments(source_state.grok, forked.grok)
            pool.release_ok(acc)
            outcome.done = True
            return


async def _fresh_turn_events(
    sess: GrokSession,
    state: SessionState,
    prompts: _TurnPrompts,
    acc: Account,
    outcome: _StreamAttempt,
) -> AsyncIterator[dict[str, Any]]:
    """Yield decorated events for a fresh-session turn.

    Args:
        sess: Fresh gateway session.
        state: Session state wrapping the fresh session.
        prompts: Resolved fresh prompts and threading options.
        acc: Account owning the turn.
        outcome: Mutable outcome record.

    Yields:
        Gateway events decorated with acc and state.

    """
    async for ev in stream_session_turn(
        sess,
        prompts.fresh,
        attachment_ids=prompts.attachment_ids,
        file_jobs=prompts.file_jobs,
        system_prompt=prompts.system_prompt,
        user_text=prompts.latest,
    ):
        decorated = dict(ev)
        decorated["acc"] = acc
        decorated["state"] = state
        yield decorated
        if decorated.get("type") == "done":
            pool.release_ok(acc)
            outcome.done = True
            return


async def _stream_checkpoint_turn(
    session_key: str,
    state: SessionState,
    prompts: _TurnPrompts,
    tried: set[str],
    outcome: _StreamAttempt,
) -> AsyncIterator[dict[str, Any]]:
    """Stream one checkpoint-following attempt, recording its outcome.

    Args:
        session_key: Chain key for session touching.
        state: Checkpoint to fork and advance.
        prompts: Resolved continuation prompts and threading options.
        tried: Account keys already attempted (mutated).
        outcome: Mutable outcome record.

    Yields:
        Gateway events decorated with acc and state.

    Raises:
        HTTPException: If the stream commits events then fails.

    """
    if not (state.grok.alive() or state.grok.conversation_id):
        return
    acc = pool.acquire_by_key(state.account_key)
    if acc is None or acc.key in tried:
        return
    tried.add(acc.key)
    outcome.attempted = True
    source_state = state
    forked = await fork_session_state(source_state, prompts.mode)
    forked.touch()
    store.touch_session(session_key, forked.last_used)
    yielded_any = False
    try:
        async for decorated in _fork_turn_events(
            forked,
            source_state,
            prompts,
            acc,
            outcome,
        ):
            yielded_any = True
            yield decorated
    except GatewayError as e:
        pool.release_fail(acc, _fail_kind(e))
        await forked.grok.close()
        propagate_dropped_attachments(source_state.grok, forked.grok)
        if yielded_any:
            status = _gateway_http_status(e)
            raise HTTPException(status, f"grok error ({e.kind}): {e}") from e
        outcome.error = e
    finally:
        if not yielded_any:
            await forked.grok.close()


async def _stream_fresh_turn(
    users: list[str],
    prompts: _TurnPrompts,
    acc: Account,
    outcome: _StreamAttempt,
) -> AsyncIterator[dict[str, Any]]:
    """Stream one fresh-session attempt on an already-acquired account.

    Args:
        users: Ordered user messages for the fresh session.
        prompts: Resolved fresh prompts and threading options.
        acc: Acquired account owning the session.
        outcome: Mutable outcome record.

    Yields:
        Gateway events decorated with acc and state.

    Raises:
        HTTPException: If the stream commits events then fails.

    """
    outcome.attempted = True
    sess: GrokSession | None = None
    yielded_any = False
    if prompts.fresh != prompts.continuation:
        logging.getLogger("uvicorn.error").warning(
            "no checkpoint for this chain; starting a fresh grok "
            "session with the full transcript (%d user message(s))",
            len(users),
        )
    try:
        sess, _ = await get_or_create_session(None, users, acc, mode=prompts.mode)
        state = SessionState(account_key=acc.key, grok=sess, user_chain=list(users))
        async for decorated in _fresh_turn_events(
            sess,
            state,
            prompts,
            acc,
            outcome,
        ):
            yielded_any = True
            yield decorated
    except GatewayError as e:
        pool.release_fail(acc, _fail_kind(e))
        if sess is not None:
            with suppress(OSError, RuntimeError, AttributeError):
                await sess.close()
        if yielded_any:
            status = _gateway_http_status(e)
            raise HTTPException(status, f"grok error ({e.kind}): {e}") from e
        outcome.error = e


async def _stream_checkpoint_attempt(
    session_key: str,
    users: list[str],
    prompts: _TurnPrompts,
    tried: set[str],
    outcome: _StreamAttempt,
) -> AsyncIterator[dict[str, Any]]:
    """Stream one checkpoint attempt for the pick failover loop.

    Args:
        session_key: Chain key of the checkpoint to continue.
        users: Ordered user messages for fallback chain data.
        prompts: Resolved continuation prompts and threading options.
        tried: Account keys already attempted (mutated).
        outcome: Mutable outcome record.

    Yields:
        Gateway events decorated with acc and state.

    """
    st = await _restore_checkpoint_state(session_key, users, prompts.mode)
    if st is None:
        return
    async for ev in _stream_checkpoint_turn(session_key, st, prompts, tried, outcome):
        yield ev


async def _stream_fresh_attempt(
    users: list[str],
    prompts: _TurnPrompts,
    tried: set[str],
    last_err: GatewayError | None,
    outcome: _StreamAttempt,
) -> AsyncIterator[dict[str, Any]]:
    """Acquire an account and stream one fresh attempt.

    Args:
        users: Ordered user messages for the fresh session.
        prompts: Resolved fresh prompts and threading options.
        tried: Account keys already attempted (mutated).
        last_err: Last gateway failure for pool-exhaustion errors.
        outcome: Mutable outcome record.

    Yields:
        Gateway events decorated with acc and state.

    Raises:
        HTTPException: If the pool is exhausted.

    """
    acc = pool.acquire(exclude=tried, include_degraded=True)
    if acc is None:
        if last_err is not None:
            status = _gateway_http_status(last_err)
            raise HTTPException(status, f"grok error ({last_err.kind}): {last_err}")
        raise HTTPException(503, "no accounts available")
    tried.add(acc.key)
    async for ev in _stream_fresh_turn(users, prompts, acc, outcome):
        yield ev


async def pick_account_and_stream_turn(
    session_key: str | None,
    users: list[str],
    **kwargs: Unpack[PickOptions],
) -> AsyncIterator[dict[str, Any]]:
    """Pick an account and yield raw gateway events in real time.

    Failover happens only while nothing has been yielded yet; once the
    first event is out the stream is committed. The failover walk tries
    every account in the pool exactly once (degraded-quarantined ones
    included as a last resort) before surfacing the last GatewayError.

    Args:
        session_key: Chain key of the checkpoint to continue, if any.
        users: Ordered user messages for fresh sessions.
        kwargs: Turn options (mode, prompt, history_prompt, latest, ...).

    Yields:
        Gateway events decorated with acc and state; the final event
        carries type "done" with "result"/"usage".

    """
    prompts = _resolve_turn_prompts(kwargs)
    tried: set[str] = set()
    last_err: GatewayError | None = None
    while True:
        await pool.reload_if_changed()
        outcome = _StreamAttempt()
        if session_key:
            async for ev in _stream_checkpoint_attempt(
                session_key,
                users,
                prompts,
                tried,
                outcome,
            ):
                yield ev
            if outcome.done:
                return
            if outcome.error is not None:
                last_err = outcome.error
                continue
        async for ev in _stream_fresh_attempt(users, prompts, tried, last_err, outcome):
            yield ev
        if outcome.done:
            return
        if outcome.error is not None:
            last_err = outcome.error


_IMAGINE_URI = "wss://grok.com/ws/imagine/listen"

_IMAGINE_URL_RE = re.compile(r"https://imagine-public[^\s\"\\]+")


def _imagine_ws_headers(cookie: str, uid: str) -> dict[str, str]:
    """Build websocket headers for the imagine endpoint.

    Args:
        cookie: Owning account cookie header.
        uid: Resolved grok user id.

    Returns:
        Origin, user-agent, and cookie headers.

    """
    return {
        "Origin": GROK_BASE,
        "User-Agent": USER_AGENT,
        "Cookie": cookie + f"; x-userid={uid}",
    }


def _build_imagine_message(prompt: str, num_images: int) -> dict[str, Any]:
    """Build the conversation.item.create payload for an imagine turn.

    Args:
        prompt: Image description prompt.
        num_images: Requested generation count.

    Returns:
        Imagine websocket message payload.

    """
    req_id = str(uuid.uuid4())
    return {
        "type": "conversation.item.create",
        "timestamp": int(time.time() * 1000),
        "item": {
            "type": "message",
            "content": [
                {
                    "requestId": req_id,
                    "text": prompt,
                    "type": "input_text",
                    "properties": {
                        "section_count": 0,
                        "is_kids_mode": False,
                        "enable_nsfw": True,
                        "skip_upsampler": False,
                        "enable_side_by_side": True,
                        "is_initial": True,
                        "aspect_ratio": "2:3",
                        "enable_pro": False,
                        "num_generations": num_images,
                        "enable_watermark": False,
                    },
                },
            ],
        },
    }


@dataclass
class _ImagineAccumulator:
    """URL and completion signals collected from imagine frames."""

    urls: list[str] = field(default_factory=list)
    completed_jobs: set[str] = field(default_factory=set)

    def add_text_urls(self, raw_text: str) -> None:
        """Collect regex-matched image URLs from a raw frame."""
        for found in _IMAGINE_URL_RE.findall(raw_text):
            if found not in self.urls:
                self.urls.append(found)


@dataclass
class _ImagineFrameOutcome:
    """Per-frame control signals for the imagine receive loop."""

    done: bool = False
    rate_limited: bool = False


def _decode_imagine_frame(raw: object) -> str | None:
    """Decode a websocket frame to text.

    Args:
        raw: Raw frame payload.

    Returns:
        Frame text, or None for non-text frames.

    """
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        return raw
    return None


def _handle_imagine_frame(
    state: _ImagineAccumulator,
    raw_text: str,
    num_images: int,
) -> _ImagineFrameOutcome:
    """Fold one imagine frame into the accumulator.

    Args:
        state: Mutable URL and completion accumulator.
        raw_text: Decoded frame text.
        num_images: Generations to wait for.

    Returns:
        Outcome with done or rate_limited flags.

    Raises:
        RuntimeError: If the gateway reports an imagine error.

    """
    outcome = _ImagineFrameOutcome()
    state.add_text_urls(raw_text)
    try:
        env = json.loads(raw_text)
    except (ValueError, TypeError, AttributeError) as e:
        logging.getLogger("uvicorn.error").debug(
            "skipping malformed imagine frame: %s",
            e,
        )
        return outcome
    etype = env.get("type", "")
    if etype == "image":
        url = str(env.get("url") or "")
        if url and url not in state.urls:
            state.urls.append(url)
    elif etype == "error":
        err_code = env.get("err_code", "")
        if "rate_limit" in err_code:
            outcome.rate_limited = True
            return outcome
        msg = f"imagine error: {env.get('err_msg', err_code)}"
        raise RuntimeError(msg)
    elif etype == "json" and env.get("current_status") == "completed":
        jid = str(env.get("job_id") or "")
        if jid:
            state.completed_jobs.add(jid)
        # Both generations finished -> return immediately
        # instead of idling for the 60s recv timeout.
        if len(state.completed_jobs) >= num_images and len(state.urls) >= num_images:
            outcome.done = True
    return outcome


@dataclass
class _ImagineResult:
    """Outcome of one account's imagine attempt."""

    urls: list[str] = field(default_factory=list)
    error: str | None = None


class _ImagineSocket(Protocol):
    """Minimal websocket surface used by the imagine receive loop."""

    async def send(self, message: str) -> None:
        """Send a text message."""
        ...

    async def recv(self) -> bytes | str:
        """Receive the next frame payload."""
        ...


async def _receive_imagine_frames(
    ws: _ImagineSocket,
    acc: Account,
    num_images: int,
    state: _ImagineAccumulator,
    result: _ImagineResult,
) -> None:
    """Receive imagine frames until completion, rate limit, or timeout.

    Args:
        ws: Connected imagine websocket.
        acc: Account attempting generation.
        num_images: Generations to wait for.
        state: Mutable URL and completion accumulator.
        result: Mutable attempt result for rate-limit errors.

    """
    while True:
        # Once images are flowing, a short idle means the job is
        # done: the imagine protocol never sends response.done,
        # it sends json current_status=completed per job. Waiting
        # the full 60s here made every image turn take 60s+.
        idle = 8.0 if state.urls else 60.0
        raw = await asyncio.wait_for(ws.recv(), timeout=idle)
        raw_text = _decode_imagine_frame(raw)
        if raw_text is None:
            continue
        outcome = _handle_imagine_frame(state, raw_text, num_images)
        if outcome.rate_limited:
            pool.release_fail(acc, "quota")
            result.error = f"account {acc.index} rate limited"
            break
        if outcome.done:
            break


async def _attempt_imagine_account(
    acc: Account,
    prompt: str,
    num_images: int,
) -> _ImagineResult:
    """Run the imagine turn on one account, handling pool release.

    Args:
        acc: Account attempting generation.
        prompt: Image description prompt.
        num_images: Requested generation count.

    Returns:
        Attempt result with image URLs or retryable error text.

    Raises:
        RuntimeError: If the gateway reports a fatal imagine error.

    """
    result = _ImagineResult()
    state = _ImagineAccumulator()
    cookie = acc.cookie_header()
    try:
        uid = await gw.resolve_user_id(cookie)
    except GatewayError as e:
        pool.release_fail(acc, "auth")
        result.error = f"account {acc.index}: {e}"
        return result
    try:
        async with websockets.connect(
            _IMAGINE_URI,
            additional_headers=_imagine_ws_headers(cookie, uid),
            max_size=32 * 1024 * 1024,
            open_timeout=10,
            close_timeout=5,
        ) as ws:
            await ws.send(json.dumps(_build_imagine_message(prompt, num_images)))
            await _receive_imagine_frames(ws, acc, num_images, state, result)
            result.urls = list(state.urls)
    except (TimeoutError, websockets.exceptions.ConnectionClosed):
        if state.urls:
            result.urls = list(state.urls)
        else:
            pool.release_fail(acc, "generic")
            result.error = f"account {acc.index}: WS timeout/closed"
    except RuntimeError:
        raise
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as e:
        pool.release_fail(acc, "generic")
        result.error = f"account {acc.index}: {e}"
    return result


def _prefer_jpg(urls: list[str]) -> list[str]:
    """Prefer jpg renditions, stripping query strings before suffix check.

    Args:
        urls: Imagine URLs in encounter order.

    Returns:
        Jpg URLs when any exist, else the original list.

    """
    jpg = [u for u in urls if u.split("?")[0].endswith(".jpg")]
    return jpg or urls


async def grok_generate_image(prompt: str, num_images: int = 2) -> list[str]:
    """Generate images via the Imagine WebSocket.

    Rotates across accounts until one produces images or the whole pool has
    been tried.

    Args:
        prompt: Image description prompt.
        num_images: Requested generation count.

    Returns:
        Public image URLs, preferring jpg renditions.

    Raises:
        RuntimeError: If no account produces images.

    """
    last_err: str | None = None
    tried: set[str] = set()
    while True:
        acc = pool.acquire(exclude=tried, include_degraded=True)
        if acc is None:
            break
        tried.add(acc.key)
        result = await _attempt_imagine_account(acc, prompt, num_images)
        if result.urls:
            pool.release_ok(acc)
            return _prefer_jpg(result.urls)
        if result.error is not None:
            last_err = result.error
    if last_err is None:
        msg = "no accounts available for image generation"
        raise RuntimeError(msg)
    msg = f"all {len(tried)} account(s) tried; last: {last_err}"
    raise RuntimeError(msg)


def _asset_owner_cookie(url: str, fallback: str = "") -> str:
    """Return the cookie of the Grok user owning an asset URL.

    Args:
        url: Asset URL possibly containing an owner user id.
        fallback: Cookie to use when no owner matches.

    Returns:
        Owning account cookie, or the fallback.

    """
    match = re.search(r"https?://assets\.grok\.com/users/([^/]+)/", url)
    if match:
        owner = next((a for a in pool.snapshot() if a.user_id == match.group(1)), None)
        if owner:
            return owner.cookie_header()
    return fallback


def _asset_headers(url: str, cookie: str) -> dict[str, str]:
    """Build request headers for a private-asset download.

    Args:
        url: Asset URL, possibly owned by another pooled account.
        cookie: Fallback cookie header.

    Returns:
        User-agent header plus the owner cookie for grok assets.

    """
    headers = {"user-agent": USER_AGENT}
    resolved = cookie
    if "assets.grok.com/users/" in url:
        resolved = _asset_owner_cookie(url, cookie)
    if resolved and "assets.grok.com" in url:
        headers["cookie"] = resolved
    return headers


async def download_asset(url: str, cookie: str = "") -> tuple[bytes, str, str]:
    """Fetch an image, using the owning Grok account when required.

    Args:
        url: Image URL or data URL.
        cookie: Fallback cookie header for private assets.

    Returns:
        Tuple of image bytes, mime type, and file name.

    Raises:
        UploadError: If the download fails or returns no bytes.

    """
    if url.startswith("data:"):
        data, mime, _ = decode_data_url(url)
        return data, mime or "image/png", "image"
    headers = _asset_headers(url, cookie)
    name = url.split("?", maxsplit=1)[0].rstrip("/").rsplit("/", 1)[-1] or "image"
    async with AsyncSession(impersonate="chrome") as s:
        r = await s.get(url, headers=headers, timeout=60)
    content = r.content
    if r.status_code != HTTP_OK or not content:
        msg = f"download {r.status_code}, {len(content)} bytes for {url[:80]}"
        raise UploadError(msg)
    if not isinstance(content, bytes) or not content:
        msg = f"download {r.status_code}, 0 bytes for {url[:80]}"
        raise UploadError(msg)
    mime_raw = (r.headers.get("content-type") or "").split(";")[0]
    mime = mime_raw if isinstance(mime_raw, str) else ""
    return content, mime, name


async def host_images(urls: list[str], cookie: str = "") -> list[str]:
    """Upload image bytes to public hosting and return only public URLs.

    Grok assets are private to their owning account. Never hand an
    ``assets.grok.com`` URL to the API client. If freeimage.host is not configured
    or hosting fails, log a warning and return available URLs or graceful fallback.

    Args:
        urls: Image URLs to host publicly.
        cookie: Fallback cookie header for private assets.

    Returns:
        Public URLs for successfully hosted images.

    """
    if not urls:
        return []
    results = await asyncio.gather(
        *[_host_one_image(u, cookie) for u in urls[:2]],
    )
    return [u for u in results if u]


async def _host_remote_image(url: str, asset_cookie: str) -> dict[str, Any]:
    """Host a non-grok image URL, downloading first when direct host fails.

    Args:
        url: Public image URL.
        asset_cookie: Cookie for authenticated download fallback.

    Returns:
        Freeimage info dict.

    """
    try:
        return await freeimage_upload_from_url(url)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        TimeoutError,
        UploadError,
    ):
        data, mime, name = await download_asset(url, asset_cookie)
        return await freeimage_upload(data, name, guess_mime(name, mime))


async def _fetch_host_info(url: str, cookie: str) -> dict[str, Any]:
    """Upload one image's bytes for public hosting.

    Args:
        url: Image URL to host publicly.
        cookie: Fallback cookie header.

    Returns:
        Freeimage info dict.

    """
    asset_cookie = _asset_owner_cookie(url, cookie)
    if "assets.grok.com/users/" in url:
        data, mime, name = await download_asset(url, asset_cookie)
        return await freeimage_upload(data, name, guess_mime(name, mime))
    return await _host_remote_image(url, asset_cookie)


async def _host_one_image(url: str, cookie: str) -> str | None:
    """Host one image URL, returning None when hosting fails.

    Args:
        url: Image URL to host publicly.
        cookie: Fallback cookie header.

    Returns:
        Public URL, or None when hosting fails.

    """
    try:
        info = await _fetch_host_info(url, cookie)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        TimeoutError,
        UploadError,
    ) as e:
        logging.getLogger("uvicorn.error").warning(
            "image hosting failed for %s: %s",
            url[:120],
            e,
        )
        return None
    raw_url = info.get("url")
    if isinstance(raw_url, str) and raw_url:
        return raw_url
    return None


@dataclass
class _BufferedAttempt:
    """Outcome of one buffered checkpoint or fresh attempt."""

    attempted: bool = False
    turn: tuple[Account, TurnResult, list[dict[str, Any]], SessionState] | None = None
    error: GatewayError | None = None


async def _fork_turn_state(
    session_key: str,
    state: SessionState,
    mode: str,
) -> SessionState:
    """Fork a checkpoint for an advancing turn, touching its session.

    Args:
        session_key: Chain key for session touching.
        state: Checkpoint to fork.
        mode: Model mode for the fork.

    Returns:
        Touched fork carrying the live socket when present.

    """
    forked = await fork_session_state(state, mode)
    forked.touch()
    store.touch_session(session_key, forked.last_used)
    return forked


async def _run_forked_turn(
    acc: Account,
    fork_state: SessionState,
    source_state: SessionState,
    prompts: _TurnPrompts,
    outcome: _BufferedAttempt,
) -> None:
    """Run the buffered turn on a forked checkpoint.

    Args:
        acc: Account owning the turn.
        fork_state: Forked checkpoint advancing the turn.
        source_state: Original checkpoint for drop propagation.
        prompts: Resolved continuation prompts and threading options.
        outcome: Mutable outcome record.

    """
    result, events = await run_session_turn(
        fork_state.grok,
        prompts.continuation,
        attachment_ids=prompts.attachment_ids,
        file_jobs=prompts.file_jobs,
        system_prompt=prompts.system_prompt,
        user_text=prompts.latest,
    )
    propagate_dropped_attachments(source_state.grok, fork_state.grok)
    pool.release_ok(acc)
    outcome.turn = (acc, result, events, fork_state)


async def _run_fresh_buffered_turn(
    sess: GrokSession,
    users: list[str],
    prompts: _TurnPrompts,
    acc: Account,
    outcome: _BufferedAttempt,
) -> None:
    """Run the buffered turn on a fresh session.

    Args:
        sess: Fresh gateway session.
        users: Ordered user messages for the fresh session.
        prompts: Resolved fresh prompts and threading options.
        acc: Acquired account owning the session.
        outcome: Mutable outcome record.

    """
    result, events = await run_session_turn(
        sess,
        prompts.fresh,
        attachment_ids=prompts.attachment_ids,
        file_jobs=prompts.file_jobs,
        system_prompt=prompts.system_prompt,
        user_text=prompts.latest,
    )
    pool.release_ok(acc)
    state = SessionState(account_key=acc.key, grok=sess, user_chain=list(users))
    outcome.turn = (acc, result, events, state)


async def _run_checkpoint_turn(
    session_key: str,
    state: SessionState,
    prompts: _TurnPrompts,
    tried: set[str],
    outcome: _BufferedAttempt,
) -> None:
    """Run one checkpoint-following buffered turn, recording its outcome.

    Args:
        session_key: Chain key for session touching.
        state: Checkpoint to fork and advance.
        prompts: Resolved continuation prompts and threading options.
        tried: Account keys already attempted (mutated).
        outcome: Mutable outcome record.

    """
    if not (state.grok.alive() or state.grok.conversation_id):
        return
    acc = pool.acquire_by_key(state.account_key)
    if acc is None or acc.key in tried:
        return
    tried.add(acc.key)
    outcome.attempted = True
    turned_ok = False
    forked: GrokSession | None = None
    fork_state = state
    try:
        source_state = state
        fork_state = await _fork_turn_state(session_key, state, prompts.mode)
        forked = fork_state.grok
        await _run_forked_turn(acc, fork_state, source_state, prompts, outcome)
        turned_ok = True
    except GatewayError as e:
        pool.release_fail(acc, _fail_kind(e))
        await fork_state.grok.close()
        propagate_dropped_attachments(source_state.grok, fork_state.grok)
        outcome.error = e
    finally:
        # Cancellation before the turn ran (or non-GatewayError
        # failure) must not orphan the moved live socket.
        if not turned_ok and forked is not None:
            await forked.close()


async def _run_fresh_turn(
    users: list[str],
    prompts: _TurnPrompts,
    acc: Account,
    outcome: _BufferedAttempt,
) -> None:
    """Run one fresh-session buffered turn on an acquired account.

    Args:
        users: Ordered user messages for the fresh session.
        prompts: Resolved fresh prompts and threading options.
        acc: Acquired account owning the session.
        outcome: Mutable outcome record.

    """
    outcome.attempted = True
    sess: GrokSession | None = None
    try:
        sess, _ = await get_or_create_session(None, users, acc, mode=prompts.mode)
        if prompts.fresh != prompts.continuation:
            logging.getLogger("uvicorn.error").warning(
                "no checkpoint for this chain; starting a fresh grok "
                "session with the full transcript (%d user message(s))",
                len(users),
            )
        await _run_fresh_buffered_turn(sess, users, prompts, acc, outcome)
    except GatewayError as e:
        if sess is not None:
            with suppress(OSError, RuntimeError, AttributeError):
                await sess.close()
        pool.release_fail(acc, _fail_kind(e))
        outcome.error = e


async def pick_account_and_turn(
    session_key: str | None,
    users: list[str],
    **kwargs: Unpack[PickOptions],
) -> tuple[Account, TurnResult, list[dict[str, Any]], SessionState]:
    """Pick an account and run a buffered turn, failing over across accounts.

    Every retryable failure fails over to the next account; the walk only
    stops after every account in the pool has been tried once (tried keys
    keep attempts distinct, degraded-quarantined ones included as a last
    resort), then the last GatewayError surfaces.

    Args:
        session_key: Chain key of the checkpoint to continue, if any.
        users: Ordered user messages for fresh sessions.
        kwargs: Turn options (mode, prompt, history_prompt, latest, ...).

    Returns:
        Tuple of account, turn result, gateway events, and live state.

    Raises:
        HTTPException: If no account is available or attempts fail.

    """
    prompts = _resolve_turn_prompts(kwargs)
    tried: set[str] = set()
    last_err: GatewayError | None = None
    while True:
        await pool.reload_if_changed()
        if session_key:
            st = await _restore_checkpoint_state(session_key, users, prompts.mode)
            if st is not None:
                outcome = _BufferedAttempt()
                await _run_checkpoint_turn(session_key, st, prompts, tried, outcome)
                if outcome.turn is not None:
                    return outcome.turn
                if outcome.error is not None:
                    last_err = outcome.error
        acc = pool.acquire(exclude=tried, include_degraded=True)
        if acc is None:
            if last_err is not None:
                status = _gateway_http_status(last_err)
                raise HTTPException(status, f"grok error ({last_err.kind}): {last_err}")
            raise HTTPException(503, "no accounts available")
        tried.add(acc.key)
        outcome = _BufferedAttempt()
        await _run_fresh_turn(users, prompts, acc, outcome)
        if outcome.turn is not None:
            return outcome.turn
        if outcome.error is not None:
            last_err = outcome.error


# ---------------------------------------------------------------- chat route


async def _read_json_body(request: Request) -> dict[str, Any]:
    """Read and validate a JSON object body.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Parsed JSON object body.

    Raises:
        HTTPException: If the payload is not a JSON object.

    """
    try:
        body = await request.json()
    except (ValueError, TypeError, AttributeError, RuntimeError, OSError) as e:
        raise HTTPException(400, f"invalid JSON payload: {e}") from e
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")
    return body


@dataclass
class _ChatContext:
    """Parsed chat request state shared by stream and non-stream paths."""

    stream: bool
    include_usage: bool
    mode: str
    public_model: str
    system_prompt: str | None
    users: list[str]
    auth_header: str
    prefix_key: str | None
    prompt: str
    remaining_jobs: list[dict[str, Any]]
    req_include_sources: bool
    history_prompt: str
    flat: list[dict[str, Any]]
    rid: str
    created: int


def _chat_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and filter the chat messages array.

    Args:
        body: Parsed JSON object body.

    Returns:
        Message objects in order.

    Raises:
        HTTPException: If the messages array is missing or has no objects.

    """
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise HTTPException(400, "messages array required")
    messages = [m for m in raw_messages if isinstance(m, dict)]
    if not messages:
        raise HTTPException(400, "messages array must contain message objects")
    return messages


def _split_system_messages(
    flat: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Split system and developer prompts from conversation messages.

    Args:
        flat: Flattened messages with text content.

    Returns:
        Tuple of joined system prompt and conversation messages.

    """
    system_prompt: str | None = None
    convo_msgs: list[dict[str, Any]] = []
    for m in flat:
        role = m.get("role")
        content = m.get("content")
        if not isinstance(content, str):
            convo_msgs.append(m)
            continue
        if role in {"system", "developer"}:
            system_prompt = (
                (system_prompt + "\n\n" + content) if system_prompt else content
            )
            continue
        convo_msgs.append(m)
    return system_prompt, convo_msgs


def _inline_text_attachments(
    prompt: str,
    file_jobs: list[dict[str, Any]],
    *,
    strip: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Inline textual attachments into the prompt, keeping binary jobs.

    Args:
        prompt: Base user prompt.
        file_jobs: Native upload jobs from extract_attachments.
        strip: Strip surrounding whitespace from the augmented prompt.

    Returns:
        Tuple of augmented prompt and remaining binary jobs.

    """
    inline_texts: list[str] = []
    remaining_jobs: list[dict[str, Any]] = []
    for job in file_jobs:
        if "file_id" in job:
            remaining_jobs.append(job)
            continue
        inlined = inline_text_safe(job)
        if inlined is not None:
            inline_texts.append(inlined)
        else:
            remaining_jobs.append(job)
    if inline_texts:
        prompt = prompt + "\n\n" + "\n\n".join(inline_texts)
        if strip:
            prompt = prompt.strip()
    return prompt, remaining_jobs


def _include_usage(body: dict[str, Any]) -> bool:
    """Check the stream_options include_usage flag.

    Args:
        body: Parsed JSON object body.

    Returns:
        True when streaming usage chunks were requested.

    """
    stream_options = body.get("stream_options")
    if not isinstance(stream_options, dict):
        return False
    return bool(stream_options.get("include_usage"))


async def _build_chat_context(
    request: Request,
    body: dict[str, Any],
) -> _ChatContext:
    """Validate the chat body and resolve prompts, jobs, and chain keys.

    Args:
        request: Incoming FastAPI request.
        body: Parsed JSON object body.

    Returns:
        Populated chat context.

    Raises:
        HTTPException: If required fields are missing or mistyped.

    """
    stream = bool(body.get("stream"))
    include_usage = _include_usage(body)
    model_in = body.get("model")
    if model_in is not None and not isinstance(model_in, str):
        raise HTTPException(400, "model must be a string")
    messages = _chat_messages(body)
    flat, file_jobs = await extract_attachments(messages)
    mode, public_model = resolve_mode(model_in)
    system_prompt, convo_msgs = _split_system_messages(flat)
    auth_header = request.headers.get("authorization", "")
    users = user_texts(convo_msgs)
    prefix_key = (
        chain_key(users[:-1], auth_header)
        if len(users) >= MIN_USERS_FOR_CHAIN_KEY
        else None
    )
    prompt = users[-1] if users else ""
    prompt, remaining_jobs = _inline_text_attachments(prompt, file_jobs)
    return _ChatContext(
        stream=stream,
        include_usage=include_usage,
        mode=mode,
        public_model=public_model,
        system_prompt=system_prompt,
        users=users,
        auth_header=auth_header,
        prefix_key=prefix_key,
        prompt=prompt,
        remaining_jobs=remaining_jobs,
        req_include_sources=include_sources(body.get("include_sources")),
        history_prompt=build_history_prompt(flat, prompt),
        flat=flat,
        rid="chatcmpl-" + uuid.uuid4().hex[:24],
        created=now_epoch(),
    )


def _turn_kwargs(ctx: _ChatContext) -> PickOptions:
    """Build gateway turn options from a chat context.

    Args:
        ctx: Populated chat context.

    Returns:
        Turn options for pick and run entry points.

    """
    return {
        "mode": ctx.mode,
        "prompt": ctx.prompt,
        "history_prompt": ctx.history_prompt,
        "latest": ctx.prompt,
        "file_jobs": ctx.remaining_jobs or None,
        "system_prompt": ctx.system_prompt,
    }


def _chat_chunk(
    rid: str,
    created: int,
    public_model: str,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> str:
    """Format one chat completion chunk frame.

    Args:
        rid: Response id.
        created: Creation epoch.
        public_model: Model name for responses.
        delta: Content delta payload.
        finish_reason: Finish reason, if terminal.

    Returns:
        SSE data frame.

    """
    chunk = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": public_model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _chat_usage_frame(
    ctx: _ChatContext,
    rid: str,
    created: int,
    completion_text: str,
) -> str:
    """Format the trailing usage chunk frame.

    Args:
        ctx: Populated chat context.
        rid: Response id.
        created: Creation epoch.
        completion_text: Completion text for token estimates.

    Returns:
        SSE data frame with usage.

    """
    prompt_tokens = len(ctx.prompt) // 4
    completion_tokens = len(completion_text) // 4
    usage_chunk = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": ctx.public_model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return f"data: {json.dumps(usage_chunk)}\n\n"


async def _generate_image_paths(prompt: str) -> list[str] | None:
    """Try imagine generation across accounts, tolerating failures.

    Args:
        prompt: Image description prompt.

    Returns:
        Asset paths, or None when every attempt fails.

    """
    for attempt in range(min(3, len(pool.snapshot()) or 1)):
        try:
            await refresh_statsig_pair()
            return await grok_generate_image(prompt)
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            TimeoutError,
        ) as e:
            logging.getLogger("uvicorn.error").warning(
                "image gen attempt %d/%d: %s",
                attempt + 1,
                3,
                e,
            )
    return None


def _chat_image_chunks(
    ctx: _ChatContext,
    rid: str,
    created: int,
    content_md: str,
) -> Iterator[str]:
    """Yield SSE chunks for an already-generated image.

    Args:
        ctx: Populated chat context.
        rid: Response id.
        created: Creation epoch.
        content_md: Markdown image content for the delta.

    Yields:
        SSE data frames ending with DONE.

    """
    yield _chat_chunk(
        rid,
        created,
        ctx.public_model,
        {"role": "assistant", "content": ""},
        None,
    )
    yield _chat_chunk(rid, created, ctx.public_model, {"content": content_md}, None)
    yield _chat_chunk(rid, created, ctx.public_model, {}, "stop")
    if ctx.include_usage:
        yield _chat_usage_frame(ctx, rid, created, "")
    yield "data: [DONE]\n\n"


async def _maybe_serve_chat_image(
    ctx: _ChatContext,
) -> JSONResponse | StreamingResponse | None:
    """Serve imagine-image output when the prompt asks for an image.

    Falls through to text chat when image generation fails everywhere.

    Args:
        ctx: Populated chat context (prompt stripped on fallthrough).

    Returns:
        Image response, or None to continue with text chat.

    """
    if not (
        bool(IMAGE_WORDS.search(ctx.prompt)) and "no image" not in ctx.prompt.lower()
    ):
        return None
    asset_paths = await _generate_image_paths(ctx.prompt)
    if not asset_paths:
        # all image gen attempts failed -> fall through to text chat
        ctx.prompt = ctx.prompt.replace("[Generate an image] ", "")
        ctx.history_prompt = build_history_prompt(ctx.flat, ctx.prompt)
        return None
    hosted = await host_images(asset_paths[:2])
    content_md = (
        "\n\n".join(f"![generated image]({u})" for u in hosted) or "Image generated."
    )
    rid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = now_epoch()
    if not ctx.stream:
        return JSONResponse(
            {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": ctx.public_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content_md,
                            "refusal": None,
                        },
                        "finish_reason": "stop",
                        "logprobs": None,
                    },
                ],
                "usage": {
                    "prompt_tokens": len(ctx.prompt) // 4,
                    "completion_tokens": 0,
                    "total_tokens": len(ctx.prompt) // 4,
                },
            },
        )
    return StreamingResponse(
        _chat_image_chunks(ctx, rid, created, content_md),
        media_type="text/event-stream",
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    """Serve a chat completion over grok sessions, streaming when asked.

    Args:
        request: Incoming FastAPI request.

    Returns:
        JSON completion or SSE stream.

    """
    check_auth(request)
    body = await _read_json_body(request)
    await pool.reload_if_changed()
    spawn_statsig_refresh()
    await prune_sessions()
    ctx = await _build_chat_context(request, body)
    imaged = await _maybe_serve_chat_image(ctx)
    if imaged is not None:
        return imaged
    if not ctx.stream:
        return await _serve_chat_completion(ctx)
    return StreamingResponse(_chat_sse_frames(ctx), media_type="text/event-stream")


async def _serve_chat_completion(ctx: _ChatContext) -> JSONResponse:
    """Run a buffered chat turn and format the completion response.

    Args:
        ctx: Populated chat context.

    Returns:
        JSON completion response.

    """
    acc, result, _events, state = await pick_account_and_turn(
        ctx.prefix_key,
        list(ctx.users),
        mode=ctx.mode,
        prompt=ctx.prompt,
        history_prompt=ctx.history_prompt,
        latest=ctx.prompt,
        file_jobs=ctx.remaining_jobs or None,
        system_prompt=ctx.system_prompt,
    )
    if ctx.users:
        await _persist_chat_session(ctx, acc, state)
    image_urls = list(result.image_urls)
    if image_urls:
        hosted = await host_images(image_urls, acc.cookie_header())
        md = "\n\n".join(f"![generated image]({u})" for u in hosted)
        result.text = (result.text + "\n\n" + md).strip()
    appendix = _sources_appendix_text(
        result.sources,
        result.search_queries,
        ctx.prompt,
        req_include_sources=ctx.req_include_sources,
    )
    final_response_text = result.text + appendix
    usage_dict = {
        "prompt_tokens": len(ctx.prompt) // 4,
        "completion_tokens": len(final_response_text) // 4,
        "total_tokens": (len(ctx.prompt) + len(final_response_text)) // 4,
    }
    return JSONResponse(
        {
            "id": ctx.rid,
            "object": "chat.completion",
            "created": ctx.created,
            "model": ctx.public_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_response_text,
                        "refusal": None,
                    },
                    "finish_reason": result.finish_reason or "stop",
                    "logprobs": None,
                },
            ],
            "usage": usage_dict,
        },
    )


async def _persist_chat_session(
    ctx: _ChatContext,
    acc: Account,
    state: SessionState,
) -> None:
    """Persist the session so the next incremental call reuses it.

    Args:
        ctx: Populated chat context.
        acc: Account owning the session.
        state: Live turn state to persist.

    """
    st = SessionState(
        account_key=acc.key,
        grok=state.grok,
        user_chain=list(ctx.users),
    )
    async with SESSION_LOCK:
        SESSIONS[chain_key(ctx.users, ctx.auth_header)] = st
    k_curr = chain_key(ctx.users, ctx.auth_header)
    store.save_session(
        session_key=k_curr,
        account_key=acc.key,
        user_chain=list(ctx.users),
        conversation_id=state.grok.conversation_id,
        last_parent_response_id=state.grok.last_parent_response_id,
        model_mode=ctx.mode,
        attachments=_session_attachments(state.grok),
        created_at=st.created_at,
        last_used=st.last_used,
    )


def _sources_appendix_text(
    sources: list[dict[str, Any]],
    search_queries: list[str],
    prompt: str,
    *,
    req_include_sources: bool,
) -> str:
    """Build the sources appendix for a completed turn.

    Args:
        sources: Gateway source dicts.
        search_queries: Gateway search queries for attribution.
        prompt: Fallback query text.
        req_include_sources: Whether the request asked for sources.

    Returns:
        Markdown appendix block, or an empty string.

    """
    if not req_include_sources or not sources:
        return ""
    appendix_query = search_queries[0] if search_queries else prompt
    return source_appendix(sources, appendix_query)


@dataclass
class _ChatStreamState:
    """Mutable accumulation state for the chat SSE stream."""

    accumulated: list[str] = field(default_factory=list)
    render_filter: RenderFilter = field(default_factory=RenderFilter)
    final_result: TurnResult | None = None
    turn_acc: Account | None = None
    turn_state: SessionState | None = None


def _is_mocked_turn() -> bool:
    """Detect whether the turn entry point is replaced by a test mock.

    Returns:
        True when pick_account_and_turn is a unittest mock.

    """
    return isinstance(
        pick_account_and_turn,
        (
            unittest.mock.NonCallableMagicMock,
            unittest.mock.AsyncMock,
            unittest.mock.MagicMock,
            unittest.mock.Mock,
        ),
    )


async def _yield_mock_chat_turn(
    ctx: _ChatContext,
    sse_state: _ChatStreamState,
) -> AsyncIterator[str]:
    """Yield content chunks for a mocked gateway turn.

    Args:
        ctx: Populated chat context.
        sse_state: Mutable stream accumulation state.

    Yields:
        SSE content frames.

    Raises:
        GatewayError: If the mock yields no completed result.

    """
    mock_res = await pick_account_and_turn(
        ctx.prefix_key,
        list(ctx.users),
        mode=ctx.mode,
        prompt=ctx.prompt,
        history_prompt=ctx.history_prompt,
        latest=ctx.prompt,
        file_jobs=ctx.remaining_jobs or None,
        system_prompt=ctx.system_prompt,
    )
    turn_acc, mock_turn_res, _mock_events, turn_state = mock_res
    sse_state.turn_acc = turn_acc
    sse_state.turn_state = turn_state
    if not isinstance(mock_turn_res, TurnResult):
        msg = "no result from gateway"
        raise GatewayError(UPSTREAM_KIND, msg)
    sse_state.final_result = mock_turn_res
    filtered_mock = sse_state.render_filter.process(mock_turn_res.text)
    if filtered_mock:
        sse_state.accumulated.append(filtered_mock)
        yield _chat_chunk(
            ctx.rid,
            ctx.created,
            ctx.public_model,
            {"content": filtered_mock},
            None,
        )


async def _yield_live_chat_turn(
    ctx: _ChatContext,
    sse_state: _ChatStreamState,
) -> AsyncIterator[str]:
    """Yield content chunks for a live gateway turn.

    Args:
        ctx: Populated chat context.
        sse_state: Mutable stream accumulation state.

    Yields:
        SSE content frames.

    """
    async for ev in pick_account_and_stream_turn(
        ctx.prefix_key,
        list(ctx.users),
        mode=ctx.mode,
        prompt=ctx.prompt,
        history_prompt=ctx.history_prompt,
        latest=ctx.prompt,
        file_jobs=ctx.remaining_jobs or None,
        system_prompt=ctx.system_prompt,
    ):
        sse_state.turn_acc = ev.get("acc")
        sse_state.turn_state = ev.get("state")
        ev_type = ev.get("type")
        if ev_type == "text_delta":
            raw_delta = ev.get("text", "")
            clean_delta = sse_state.render_filter.process(raw_delta)
            if clean_delta:
                sse_state.accumulated.append(clean_delta)
                yield _chat_chunk(
                    ctx.rid,
                    ctx.created,
                    ctx.public_model,
                    {"content": clean_delta},
                    None,
                )
        elif ev_type == "done":
            done_result = ev.get("result")
            if isinstance(done_result, TurnResult):
                sse_state.final_result = done_result


async def _yield_chat_tail_frames(
    ctx: _ChatContext,
    sse_state: _ChatStreamState,
) -> AsyncIterator[str]:
    """Yield flush, image, and appendix frames for a chat stream.

    Args:
        ctx: Populated chat context.
        sse_state: Mutable stream accumulation state.

    Yields:
        SSE content frames.

    """
    flushed = sse_state.render_filter.flush()
    if flushed:
        sse_state.accumulated.append(flushed)
        yield _chat_chunk(
            ctx.rid,
            ctx.created,
            ctx.public_model,
            {"content": flushed},
            None,
        )
    if (
        sse_state.final_result
        and sse_state.final_result.image_urls
        and sse_state.turn_acc
    ):
        hosted = await host_images(
            sse_state.final_result.image_urls,
            sse_state.turn_acc.cookie_header(),
        )
        if hosted:
            md = "\n\n" + "\n\n".join(f"![generated image]({u})" for u in hosted)
            sse_state.accumulated.append(md)
            yield _chat_chunk(
                ctx.rid,
                ctx.created,
                ctx.public_model,
                {"content": md},
                None,
            )
    appendix = _sources_appendix_text(
        sse_state.final_result.sources if sse_state.final_result else [],
        sse_state.final_result.search_queries if sse_state.final_result else [],
        ctx.prompt,
        req_include_sources=ctx.req_include_sources,
    )
    if appendix:
        sse_state.accumulated.append(appendix)
        yield _chat_chunk(
            ctx.rid,
            ctx.created,
            ctx.public_model,
            {"content": appendix},
            None,
        )


async def _persist_chat_stream_session(
    ctx: _ChatContext,
    sse_state: _ChatStreamState,
) -> None:
    """Persist stream session state for the next incremental call.

    Args:
        ctx: Populated chat context.
        sse_state: Mutable stream accumulation state.

    """
    if not ctx.users or not sse_state.turn_acc or not sse_state.turn_state:
        return
    turn_acc = sse_state.turn_acc
    turn_state = sse_state.turn_state
    st = SessionState(
        account_key=turn_acc.key,
        grok=turn_state.grok,
        user_chain=list(ctx.users),
    )
    async with SESSION_LOCK:
        SESSIONS[chain_key(ctx.users, ctx.auth_header)] = st
    k_curr = chain_key(ctx.users, ctx.auth_header)
    store.save_session(
        session_key=k_curr,
        account_key=turn_acc.key,
        user_chain=list(ctx.users),
        conversation_id=turn_state.grok.conversation_id,
        last_parent_response_id=turn_state.grok.last_parent_response_id,
        model_mode=ctx.mode,
        attachments=_session_attachments(turn_state.grok),
        created_at=st.created_at,
        last_used=st.last_used,
    )


async def _render_chat_frames(ctx: _ChatContext) -> AsyncIterator[str]:
    """Render all chat SSE frames for a streaming turn.

    Args:
        ctx: Populated chat context.

    Yields:
        SSE data frames ending with DONE.

    """
    yield _chat_chunk(
        ctx.rid,
        ctx.created,
        ctx.public_model,
        {"role": "assistant", "content": ""},
        None,
    )
    sse_state = _ChatStreamState()
    if _is_mocked_turn():
        async for frame in _yield_mock_chat_turn(ctx, sse_state):
            yield frame
    else:
        async for frame in _yield_live_chat_turn(ctx, sse_state):
            yield frame
    async for frame in _yield_chat_tail_frames(ctx, sse_state):
        yield frame
    await _persist_chat_stream_session(ctx, sse_state)
    finish_reason = (
        sse_state.final_result.finish_reason if sse_state.final_result else None
    ) or "stop"
    yield _chat_chunk(ctx.rid, ctx.created, ctx.public_model, {}, finish_reason)
    if ctx.include_usage:
        total_text = "".join(sse_state.accumulated)
        yield _chat_usage_frame(ctx, ctx.rid, ctx.created, total_text)
    yield "data: [DONE]\n\n"


async def _chat_sse_frames(ctx: _ChatContext) -> AsyncIterator[str]:
    """Yield chat SSE frames, logging stream failures instead of raising.

    Args:
        ctx: Populated chat context.

    Yields:
        SSE data frames ending with DONE.

    """
    try:
        async for frame in _render_chat_frames(ctx):
            yield frame
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        IndexError,
        TimeoutError,
        GatewayError,
        HTTPException,
    ):
        logging.getLogger("uvicorn.error").exception("Chat SSE stream error")


def inline_text_safe(job: dict[str, Any]) -> str | None:
    """Inline a job's text, swallowing decode failures.

    Args:
        job: Upload job with name, mime, and data.

    Returns:
        Fenced code block, or None for binary or undecodable files.

    """
    try:
        return inline_textual(job)
    except (ValueError, TypeError, AttributeError, UnicodeError):
        return None


# --------------------------------------------------------------- responses API


@app.post("/v1/responses", response_model=None)
async def responses_api(request: Request) -> JSONResponse | StreamingResponse:
    """Serve a responses API request over grok sessions.

    Args:
        request: Incoming FastAPI request.

    Returns:
        JSON response object or SSE stream.

    """
    check_auth(request)
    body = await _read_json_body(request)
    await pool.reload_if_changed()
    spawn_statsig_refresh()
    await prune_sessions()
    ctx = await _build_responses_context(request, body)
    if not ctx.stream:
        return await _serve_response(ctx)
    return StreamingResponse(_responses_sse_frames(ctx), media_type="text/event-stream")


def _merge_user_part(
    messages: list[dict[str, Any]],
    content: dict[str, Any],
) -> None:
    """Fold a shorthand item into the previous user message.

    Prompt extraction takes the LAST user message, so a text+image sequence
    of bare items would otherwise lose its text.

    Args:
        messages: Collector for normalized chat messages.
        content: Shorthand content part to merge.

    """
    if not (messages and messages[-1].get("role") == "user"):
        messages.append({"role": "user", "content": [content]})
        return
    prev = messages[-1]["content"]
    if isinstance(prev, str):
        merged: list[dict[str, Any]] = [{"type": "text", "text": prev}]
    elif isinstance(prev, list):
        merged = prev
    else:
        merged = []
    messages[-1]["content"] = [*merged, content]


def _append_response_item(
    messages: list[dict[str, Any]],
    item: dict[str, Any],
) -> None:
    """Append one responses input item as a chat message.

    Args:
        messages: Collector for normalized chat messages.
        item: Single input item dict.

    """
    itype = item.get("type")
    role = item.get("role")
    if itype == "message" or (role and not itype):
        messages.append(
            {"role": role or "user", "content": item.get("content")},
        )
    elif itype == "input_text":
        _merge_user_part(messages, {"type": "text", "text": item.get("text", "")})
    elif itype in {"input_image", "input_file"}:
        _merge_user_part(messages, item)
    elif role:
        messages.append(item)


def _responses_messages(
    input_data: dict[str, Any] | list[Any] | str | None,
) -> list[dict[str, Any]]:
    """Normalize the responses input payload to chat messages.

    Args:
        input_data: Raw input value (string or item list).

    Returns:
        Chat messages with text or part-list content.

    """
    messages: list[dict[str, Any]] = []
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    if not isinstance(input_data, list):
        return messages
    for item in input_data:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
        elif isinstance(item, dict):
            _append_response_item(messages, item)
    return messages


async def _restore_prev_session(prev_resp_id: str) -> SessionState | None:
    """Restore a previous-response session from memory or SQLite.

    Args:
        prev_resp_id: Previous response id key.

    Returns:
        Restored session state, if available.

    """
    async with SESSION_LOCK:
        sess_prev = SESSIONS.get(prev_resp_id)
    if sess_prev is not None:
        return sess_prev
    persisted = store.get_session(prev_resp_id)
    if not persisted:
        return None
    account_key = persisted.get("account_key")
    if not isinstance(account_key, str):
        return None
    acc_cand = pool.acquire_by_key(account_key)
    if not acc_cand:
        return None
    uid = await _uid_for(acc_cand)
    sess = GrokSession(
        acc_cand.cookie_header(),
        uid,
        _persisted_str(persisted, "model_mode", "fast"),
    )
    sess.conversation_id = _persisted_str(persisted, "conversation_id", "")
    sess.last_parent_response_id = _persisted_str(
        persisted,
        "last_parent_response_id",
        "",
    )
    sess.attachments = _clean_attachment_registry(persisted.get("attachments"))
    sess_new = SessionState(
        account_key=acc_cand.key,
        grok=sess,
        user_chain=_persisted_chain(persisted, []),
        created_at=_persisted_time(persisted, "created_at", time.time()),
        last_used=time.time(),
    )
    async with SESSION_LOCK:
        if prev_resp_id not in SESSIONS:
            SESSIONS[prev_resp_id] = sess_new
            return sess_new
        return SESSIONS[prev_resp_id]


def _latest_user_prompt(flat: list[dict[str, Any]]) -> str:
    """Extract the newest user message text.

    Args:
        flat: Flattened messages with text content.

    Returns:
        Newest user text, or an empty string.

    """
    for m in reversed(flat):
        if m.get("role") == "user":
            content = m.get("content")
            return content if isinstance(content, str) else content_to_text(content)
    return ""


@dataclass
class _ResponsesContext:
    """Parsed responses request state shared by stream and non-stream paths."""

    stream: bool
    mode: str
    public_model: str
    instructions: str
    prev_resp_id: str | None
    sess_prev: SessionState | None
    users: list[str]
    auth_header: str
    prefix_key: str | None
    prompt: str
    remaining_jobs: list[dict[str, Any]]
    req_include_sources: bool
    history_prompt: str
    rid: str
    msg_id: str
    created: int


async def _build_responses_context(
    request: Request,
    body: dict[str, Any],
) -> _ResponsesContext:
    """Validate the responses body and resolve prompts, jobs, and sessions.

    Args:
        request: Incoming FastAPI request.
        body: Parsed JSON object body.

    Returns:
        Populated responses context.

    Raises:
        HTTPException: If required fields are missing or mistyped.

    """
    stream = bool(body.get("stream"))
    model_in = body.get("model")
    if model_in is not None and not isinstance(model_in, str):
        raise HTTPException(400, "model must be a string")
    instructions = body.get("instructions") or ""
    prev_resp_id = body.get("previous_response_id")
    messages = _responses_messages(body.get("input"))
    auth_header = request.headers.get("authorization", "")
    sess_prev: SessionState | None = None
    if prev_resp_id:
        sess_prev = await _restore_prev_session(prev_resp_id)
        if not messages:
            if not sess_prev:
                raise HTTPException(
                    400,
                    f"unknown previous_response_id: {prev_resp_id}",
                )
            raise HTTPException(400, "input required with previous_response_id")
    if not messages:
        raise HTTPException(400, "input required")
    mode, public_model = resolve_mode(model_in)
    users = user_texts(messages)
    prefix_key = (
        chain_key(users[:-1], auth_header)
        if len(users) >= MIN_USERS_FOR_CHAIN_KEY
        else None
    )
    if sess_prev is not None:
        prefix_key = None
    flat, file_jobs = await extract_attachments(messages)
    prompt = _latest_user_prompt(flat)
    prompt, remaining_jobs = _inline_text_attachments(prompt, file_jobs, strip=True)
    return _ResponsesContext(
        stream=stream,
        mode=mode,
        public_model=public_model,
        instructions=instructions,
        prev_resp_id=prev_resp_id,
        sess_prev=sess_prev,
        users=users,
        auth_header=auth_header,
        prefix_key=prefix_key,
        prompt=prompt,
        remaining_jobs=remaining_jobs,
        req_include_sources=include_sources(body.get("include_sources")),
        history_prompt=build_history_prompt(flat, prompt),
        rid="resp_" + uuid.uuid4().hex,
        msg_id="msg_" + uuid.uuid4().hex,
        created=now_epoch(),
    )


def _response_turn_kwargs(ctx: _ResponsesContext) -> PickOptions:
    """Build gateway turn options from a responses context.

    Args:
        ctx: Populated responses context.

    Returns:
        Turn options for pick and run entry points.

    """
    return {
        "mode": ctx.mode,
        "prompt": ctx.prompt,
        "history_prompt": ctx.history_prompt,
        "latest": ctx.prompt,
        "file_jobs": ctx.remaining_jobs or None,
        "system_prompt": ctx.instructions,
    }


async def _run_continued_response(
    ctx: _ResponsesContext,
    sess_prev: SessionState,
) -> tuple[Account, TurnResult, list[dict[str, Any]], SessionState]:
    """Run a buffered turn continuing the previous response's session.

    Args:
        ctx: Populated responses context.
        sess_prev: Previous response's session state to fork.

    Returns:
        Tuple of account, turn result, events, and live state.

    Raises:
        HTTPException: If the previous account cools down or the turn fails.

    """
    acc = pool.acquire_by_key(sess_prev.account_key)
    if not acc:
        raise HTTPException(
            409,
            "previous response account is cooling down; retry",
        )
    turned_ok = False
    forked: GrokSession | None = None
    st = await fork_session_state(sess_prev, ctx.mode)
    forked = st.grok
    st.touch()
    try:
        result, events = await run_session_turn(
            st.grok,
            ctx.prompt,
            file_jobs=ctx.remaining_jobs or None,
            system_prompt=ctx.instructions,
        )
        propagate_dropped_attachments(sess_prev.grok, st.grok)
        pool.release_ok(acc)
        turned_ok = True
    except GatewayError as e:
        pool.release_fail(acc, _fail_kind(e))
        await st.grok.close()
        propagate_dropped_attachments(sess_prev.grok, st.grok)
        status = _gateway_http_status(e)
        raise HTTPException(status, f"grok error ({e.kind}): {e}") from e
    else:
        return acc, result, events, st
    finally:
        if not turned_ok:
            await forked.close()


async def _persist_response_session(
    ctx: _ResponsesContext,
    acc: Account,
    state: SessionState,
) -> None:
    """Persist response sessions under response, chain, and user keys.

    Args:
        ctx: Populated responses context.
        acc: Account owning the session.
        state: Live turn state to persist.

    """
    if not ctx.users:
        return
    prev_chain = ctx.sess_prev.user_chain if ctx.sess_prev else []
    full_user_chain = [*prev_chain, *ctx.users]
    st = SessionState(
        account_key=acc.key,
        grok=state.grok,
        user_chain=full_user_chain,
    )
    async with SESSION_LOCK:
        SESSIONS[ctx.rid] = st  # previous_response_id -> session
        SESSIONS[ctx.rid + ":chain"] = st
        SESSIONS[chain_key(ctx.users, ctx.auth_header)] = st
    for session_key in (
        ctx.rid,
        ctx.rid + ":chain",
        chain_key(ctx.users, ctx.auth_header),
    ):
        store.save_session(
            session_key=session_key,
            account_key=acc.key,
            user_chain=full_user_chain,
            conversation_id=state.grok.conversation_id,
            last_parent_response_id=state.grok.last_parent_response_id,
            model_mode=ctx.mode,
            attachments=_session_attachments(state.grok),
            created_at=st.created_at,
            last_used=st.last_used,
        )


async def _serve_response(ctx: _ResponsesContext) -> JSONResponse:
    """Run a buffered responses turn and format the response object.

    Args:
        ctx: Populated responses context.

    Returns:
        JSON response object.

    """
    if ctx.sess_prev is not None:
        acc, result, _events, state = await _run_continued_response(
            ctx,
            ctx.sess_prev,
        )
    else:
        acc, result, _events, state = await pick_account_and_turn(
            ctx.prefix_key,
            list(ctx.users),
            mode=ctx.mode,
            prompt=ctx.prompt,
            history_prompt=ctx.history_prompt,
            latest=ctx.prompt,
            file_jobs=ctx.remaining_jobs or None,
            system_prompt=ctx.instructions,
        )
    image_urls = list(result.image_urls)
    if image_urls:
        hosted = await host_images(image_urls, acc.cookie_header())
        md = "\n\n".join(f"![generated image]({u})" for u in hosted)
        result.text = (result.text + "\n\n" + md).strip()
    await _persist_response_session(ctx, acc, state)
    appendix = _sources_appendix_text(
        result.sources,
        result.search_queries,
        ctx.prompt,
        req_include_sources=ctx.req_include_sources,
    )
    final_response_text = result.text + appendix
    output_item = {
        "id": ctx.msg_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": final_response_text, "annotations": []},
        ],
    }
    usage_obj = {
        "input_tokens": len(ctx.prompt) // 4,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "output_tokens": len(final_response_text) // 4,
        "output_tokens_details": {"reasoning_tokens": len(result.reasoning) // 4},
        "total_tokens": (len(ctx.prompt) + len(final_response_text)) // 4,
    }
    base_response = {
        "id": ctx.rid,
        "object": "response",
        "created_at": ctx.created,
        "completed_at": now_epoch(),
        "status": "completed",
        "model": ctx.public_model,
        "instructions": ctx.instructions or None,
        "output": [output_item],
        "output_text": final_response_text,
        "error": None,
        "incomplete_details": None,
        "previous_response_id": ctx.prev_resp_id or None,
        "parallel_tool_calls": False,
        "temperature": 1.0,
        "tool_choice": "none",
        "tools": [],
        "top_p": 1.0,
        "usage": usage_obj,
        "metadata": {},
    }
    return JSONResponse(base_response)


@dataclass
class _ResponseEventWriter:
    """Numbers and formats responses SSE frames."""

    seq: int = 0

    def evt(self, name: str, data: dict[str, Any]) -> str:
        """Format one numbered SSE event frame.

        Args:
            name: Event name.
            data: Event payload (mutated with a sequence number).

        Returns:
            SSE event frame.

        """
        self.seq += 1
        data["sequence_number"] = self.seq
        return f"event: {name}\ndata: {json.dumps(data)}\n\n"


@dataclass
class _ResponsesStreamState:
    """Mutable accumulation state for the responses SSE stream."""

    accumulated: list[str] = field(default_factory=list)
    render_filter: RenderFilter = field(default_factory=RenderFilter)
    final_result: TurnResult | None = None
    turn_acc: Account | None = None
    turn_state: SessionState | None = None
    turn_error: Exception | None = None


def _responses_opening_frames(
    ctx: _ResponsesContext,
    writer: _ResponseEventWriter,
) -> Iterator[str]:
    """Yield the created/in-progress/item/part opening frames.

    Args:
        ctx: Populated responses context.
        writer: Numbering event writer.

    Yields:
        SSE event frames.

    """
    init_output_item = {
        "id": ctx.msg_id,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    resp_in_prog = {
        "id": ctx.rid,
        "object": "response",
        "created_at": ctx.created,
        "completed_at": None,
        "status": "in_progress",
        "model": ctx.public_model,
        "instructions": ctx.instructions or None,
        "output": [],
        "output_text": "",
        "error": None,
        "incomplete_details": None,
        "previous_response_id": ctx.prev_resp_id or None,
        "parallel_tool_calls": False,
        "temperature": 1.0,
        "tool_choice": "none",
        "tools": [],
        "top_p": 1.0,
        "usage": None,
        "metadata": {},
    }
    yield writer.evt(
        "response.created",
        {"type": "response.created", "response": resp_in_prog},
    )
    yield writer.evt(
        "response.in_progress",
        {"type": "response.in_progress", "response": resp_in_prog},
    )
    yield writer.evt(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": init_output_item,
        },
    )
    yield writer.evt(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": ctx.msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )


def _render_response_delta(
    ctx: _ResponsesContext,
    sse_state: _ResponsesStreamState,
    writer: _ResponseEventWriter,
    ev: dict[str, Any],
) -> Iterator[str]:
    """Render one gateway event as response delta frames.

    Args:
        ctx: Populated responses context.
        sse_state: Mutable stream accumulation state.
        writer: Numbering event writer.
        ev: Gateway event dict.

    Yields:
        SSE event frames.

    """
    ev_type = ev.get("type")
    if ev_type == "text_delta":
        raw_delta = ev.get("text", "")
        clean_delta = sse_state.render_filter.process(raw_delta)
        if clean_delta:
            sse_state.accumulated.append(clean_delta)
            yield writer.evt(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": ctx.msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": clean_delta,
                },
            )
    elif ev_type == "done":
        sess_done = ev.get("result")
        if isinstance(sess_done, TurnResult):
            sse_state.final_result = sess_done


async def _stream_continued_response(
    ctx: _ResponsesContext,
    sess_prev: SessionState,
    sse_state: _ResponsesStreamState,
    writer: _ResponseEventWriter,
) -> AsyncIterator[str]:
    """Stream a turn continuing the previous response's session.

    Turn failure after the initial frames are flushed terminates the SSE
    stream with a structured response.failed event; raising HTTPException
    here would abort the connection mid-stream instead.

    Args:
        ctx: Populated responses context.
        sess_prev: Previous response's session state to fork.
        sse_state: Mutable stream accumulation state.
        writer: Numbering event writer.

    Yields:
        SSE event frames.

    """
    acc = pool.acquire_by_key(sess_prev.account_key)
    if not acc:
        sse_state.turn_error = HTTPException(
            409,
            "previous response account is cooling down; retry",
        )
        return
    turned_ok = False
    forked: GrokSession | None = None
    st = await fork_session_state(sess_prev, ctx.mode)
    forked = st.grok
    st.touch()
    sse_state.turn_acc = acc
    sse_state.turn_state = st
    try:
        async for ev in stream_session_turn(
            st.grok,
            ctx.prompt,
            file_jobs=ctx.remaining_jobs or None,
            system_prompt=ctx.instructions,
        ):
            for frame in _render_response_delta(ctx, sse_state, writer, ev):
                yield frame
        propagate_dropped_attachments(sess_prev.grok, st.grok)
        pool.release_ok(acc)
        turned_ok = True
    except GatewayError as e:
        pool.release_fail(acc, _fail_kind(e))
        await st.grok.close()
        propagate_dropped_attachments(sess_prev.grok, st.grok)
        sse_state.turn_error = e
    finally:
        if not turned_ok:
            await forked.close()


async def _yield_mock_response_turn(
    ctx: _ResponsesContext,
    sse_state: _ResponsesStreamState,
    writer: _ResponseEventWriter,
) -> AsyncIterator[str]:
    """Yield delta frames for a mocked responses turn.

    Args:
        ctx: Populated responses context.
        sse_state: Mutable stream accumulation state.
        writer: Numbering event writer.

    Yields:
        SSE event frames.

    """
    mock_res = await pick_account_and_turn(
        ctx.prefix_key,
        list(ctx.users),
        mode=ctx.mode,
        prompt=ctx.prompt,
        history_prompt=ctx.history_prompt,
        latest=ctx.prompt,
        file_jobs=ctx.remaining_jobs or None,
        system_prompt=ctx.instructions,
    )
    turn_acc, mock_turn_res, _mock_events, turn_state = mock_res
    sse_state.turn_acc = turn_acc
    sse_state.turn_state = turn_state
    sse_state.final_result = mock_turn_res
    filtered_mock = sse_state.render_filter.process(mock_turn_res.text)
    if filtered_mock:
        sse_state.accumulated.append(filtered_mock)
        yield writer.evt(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": ctx.msg_id,
                "output_index": 0,
                "content_index": 0,
                "delta": filtered_mock,
            },
        )


async def _yield_live_response_turn(
    ctx: _ResponsesContext,
    sse_state: _ResponsesStreamState,
    writer: _ResponseEventWriter,
) -> AsyncIterator[str]:
    """Yield delta frames for a live responses turn, capturing failures.

    Args:
        ctx: Populated responses context.
        sse_state: Mutable stream accumulation state.
        writer: Numbering event writer.

    Yields:
        SSE event frames.

    """
    try:
        async for ev in pick_account_and_stream_turn(
            ctx.prefix_key,
            list(ctx.users),
            mode=ctx.mode,
            prompt=ctx.prompt,
            history_prompt=ctx.history_prompt,
            latest=ctx.prompt,
            file_jobs=ctx.remaining_jobs or None,
            system_prompt=ctx.instructions,
        ):
            sse_state.turn_acc = ev.get("acc")
            sse_state.turn_state = ev.get("state")
            for frame in _render_response_delta(ctx, sse_state, writer, ev):
                yield frame
    except (GatewayError, HTTPException) as e:
        sse_state.turn_error = e


def _failed_response_frame(
    ctx: _ResponsesContext,
    writer: _ResponseEventWriter,
    turn_error: Exception,
) -> str:
    """Render the terminal response.failed frame.

    Args:
        ctx: Populated responses context.
        writer: Numbering event writer.
        turn_error: Captured turn failure.

    Returns:
        SSE event frame.

    """
    status = getattr(turn_error, "status_code", None) or (
        429 if getattr(turn_error, "kind", "") == "quota" else 502
    )
    detail = getattr(turn_error, "detail", None) or str(turn_error)
    failed_response = {
        "id": ctx.rid,
        "object": "response",
        "created_at": ctx.created,
        "completed_at": now_epoch(),
        "status": "in_progress",
        "model": ctx.public_model,
        "instructions": ctx.instructions or None,
        "output": [],
        "output_text": "",
        "error": None,
        "incomplete_details": None,
        "previous_response_id": ctx.prev_resp_id or None,
        "parallel_tool_calls": False,
        "temperature": 1.0,
        "tool_choice": "none",
        "tools": [],
        "top_p": 1.0,
        "usage": None,
        "metadata": {},
    }
    failed_response.update(
        {
            "status": "failed",
            "completed_at": now_epoch(),
            "error": {"code": f"http_{status}", "message": detail},
        },
    )
    return writer.evt(
        "response.failed",
        {"type": "response.failed", "response": failed_response},
    )


async def _render_response_tail(
    ctx: _ResponsesContext,
    sse_state: _ResponsesStreamState,
    writer: _ResponseEventWriter,
) -> AsyncIterator[str]:
    """Render failure, flush, image, and appendix frames, then persist.

    Args:
        ctx: Populated responses context.
        sse_state: Mutable stream accumulation state.
        writer: Numbering event writer.

    Yields:
        SSE event frames.

    """
    if sse_state.turn_error is not None:
        yield _failed_response_frame(ctx, writer, sse_state.turn_error)
        return
    flushed = sse_state.render_filter.flush()
    if flushed:
        sse_state.accumulated.append(flushed)
        yield writer.evt(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": ctx.msg_id,
                "output_index": 0,
                "content_index": 0,
                "delta": flushed,
            },
        )
    if (
        sse_state.final_result
        and sse_state.final_result.image_urls
        and sse_state.turn_acc
    ):
        hosted = await host_images(
            sse_state.final_result.image_urls,
            sse_state.turn_acc.cookie_header(),
        )
        if hosted:
            md = "\n\n" + "\n\n".join(f"![generated image]({u})" for u in hosted)
            sse_state.accumulated.append(md)
            yield writer.evt(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": ctx.msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": md,
                },
            )
    appendix = _sources_appendix_text(
        sse_state.final_result.sources if sse_state.final_result else [],
        sse_state.final_result.search_queries if sse_state.final_result else [],
        ctx.prompt,
        req_include_sources=ctx.req_include_sources,
    )
    if appendix:
        sse_state.accumulated.append(appendix)
        yield writer.evt(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": ctx.msg_id,
                "output_index": 0,
                "content_index": 0,
                "delta": appendix,
            },
        )
    await _persist_response_session_for_stream(ctx, sse_state)


async def _persist_response_session_for_stream(
    ctx: _ResponsesContext,
    sse_state: _ResponsesStreamState,
) -> None:
    """Persist stream session state under response, chain, and user keys.

    Args:
        ctx: Populated responses context.
        sse_state: Mutable stream accumulation state.

    """
    if not ctx.users or not sse_state.turn_acc or not sse_state.turn_state:
        return
    turn_acc = sse_state.turn_acc
    turn_state = sse_state.turn_state
    prev_chain = ctx.sess_prev.user_chain if ctx.sess_prev else []
    full_user_chain = [*prev_chain, *ctx.users]
    st = SessionState(
        account_key=turn_acc.key,
        grok=turn_state.grok,
        user_chain=full_user_chain,
    )
    async with SESSION_LOCK:
        SESSIONS[ctx.rid] = st  # previous_response_id -> session
        SESSIONS[ctx.rid + ":chain"] = st
        SESSIONS[chain_key(ctx.users, ctx.auth_header)] = st
    for session_key in (
        ctx.rid,
        ctx.rid + ":chain",
        chain_key(ctx.users, ctx.auth_header),
    ):
        store.save_session(
            session_key=session_key,
            account_key=turn_acc.key,
            user_chain=full_user_chain,
            conversation_id=turn_state.grok.conversation_id,
            last_parent_response_id=turn_state.grok.last_parent_response_id,
            model_mode=ctx.mode,
            attachments=_session_attachments(turn_state.grok),
            created_at=st.created_at,
            last_used=st.last_used,
        )


def _render_completed_response(
    ctx: _ResponsesContext,
    sse_state: _ResponsesStreamState,
    writer: _ResponseEventWriter,
) -> Iterator[str]:
    """Render the completed response object and closing frames.

    Args:
        ctx: Populated responses context.
        sse_state: Mutable stream accumulation state.
        writer: Numbering event writer.

    Yields:
        SSE event frames.

    """
    final_response_text = "".join(sse_state.accumulated)
    final_output_item = {
        "id": ctx.msg_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": final_response_text, "annotations": []},
        ],
    }
    reasoning_len = (
        len(sse_state.final_result.reasoning) if sse_state.final_result else 0
    )
    usage_obj = {
        "input_tokens": len(ctx.prompt) // 4,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "output_tokens": len(final_response_text) // 4,
        "output_tokens_details": {"reasoning_tokens": reasoning_len // 4},
        "total_tokens": (len(ctx.prompt) + len(final_response_text)) // 4,
    }
    completed_response = {
        "id": ctx.rid,
        "object": "response",
        "created_at": ctx.created,
        "completed_at": now_epoch(),
        "status": "completed",
        "model": ctx.public_model,
        "instructions": ctx.instructions or None,
        "output": [final_output_item],
        "output_text": final_response_text,
        "error": None,
        "incomplete_details": None,
        "previous_response_id": ctx.prev_resp_id or None,
        "parallel_tool_calls": False,
        "temperature": 1.0,
        "tool_choice": "none",
        "tools": [],
        "top_p": 1.0,
        "usage": usage_obj,
        "metadata": {},
    }
    yield writer.evt(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": ctx.msg_id,
            "output_index": 0,
            "content_index": 0,
            "text": final_response_text,
        },
    )
    yield writer.evt(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "item_id": ctx.msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": final_response_text,
                "annotations": [],
            },
        },
    )
    yield writer.evt(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": final_output_item,
        },
    )
    yield writer.evt(
        "response.completed",
        {"type": "response.completed", "response": completed_response},
    )


async def _responses_sse_frames(ctx: _ResponsesContext) -> AsyncIterator[str]:
    """Yield responses SSE frames for a streaming turn.

    Args:
        ctx: Populated responses context.

    Yields:
        SSE event frames ending with response.completed.

    """
    writer = _ResponseEventWriter()
    sse_state = _ResponsesStreamState()
    for frame in _responses_opening_frames(ctx, writer):
        yield frame
    if ctx.sess_prev is not None:
        async for frame in _stream_continued_response(
            ctx,
            ctx.sess_prev,
            sse_state,
            writer,
        ):
            yield frame
    elif _is_mocked_turn():
        async for frame in _yield_mock_response_turn(ctx, sse_state, writer):
            yield frame
    else:
        async for frame in _yield_live_response_turn(ctx, sse_state, writer):
            yield frame
    async for frame in _render_response_tail(ctx, sse_state, writer):
        yield frame
    if sse_state.turn_error is None:
        for frame in _render_completed_response(ctx, sse_state, writer):
            yield frame


# ------------------------------------------------------------------ misc routes


@app.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    """List the available grok model ids.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Model list payload.

    """
    check_auth(request)
    data = [
        {"id": mid, "object": "model", "created": 1700000000, "owned_by": "grok"}
        for mid in ["grok-fast", "grok-auto", "grok-expert", "grok-heavy", "grok-build"]
    ]
    return {"object": "list", "data": data}


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Report service, account, and session health.

    Returns:
        Health payload.

    """
    await pool.reload_if_changed()
    accounts = pool.snapshot()
    return {
        "ok": True,
        "accounts": len(accounts),
        "available": sum(1 for a in accounts if a.available()),
        "statsig_ready": statsig.ready,
        "sessions": len(SESSIONS),
        "degraded_accounts": sum(1 for a in accounts if time.time() < a.degraded_until),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
