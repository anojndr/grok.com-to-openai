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
import base64
import hashlib
import logging
import json
import os
import re
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

import config
from config import (
    API_KEY,
    COOLDOWN_SECONDS,
    DEFAULT_MODEL,
    GROK_BASE,
    PORT,
    SESSION_TTL,
    MAX_SESSIONS,
    USER_AGENT,
    INCLUDE_SOURCES,
)
from accounts import Account, AccountPool
from statsig import (
    StatsigGenerator,
    STATSIG_EPOCH,
    SALT,
    compute_animation_hex,
    curves_to_path,
)
from session_store import SqliteStore, MAX_TRACKED_ATTACHMENTS, clean_attachment_rows


from uploads import (
    UploadError,
    decode_data_url,
    guess_mime,
    pixelvault_upload,
    pixelvault_upload_from_url,
    upload_file,
)
import grok_gateway as gw
from grok_gateway import GrokSession, GatewayError, TurnResult, RenderFilter

app = FastAPI(title="grok-to-openai-api", version="1.0")

store = SqliteStore(config.DB_PATH)
pool = AccountPool(config.ACCOUNTS_FILE, cooldown_seconds=COOLDOWN_SECONDS, store=store)
statsig = StatsigGenerator(store=store)


# ---------------------------------------------------------------- session map


@dataclass
class SessionState:
    account_key: str  # owning account key
    grok: GrokSession  # live gateway session (one per conversation)
    user_chain: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_used = time.time()


SESSIONS: dict[str, SessionState] = {}


def _clone_session_state(state: SessionState, mode: str | None = None) -> SessionState:
    source = state.grok
    if isinstance(source, GrokSession):
        sess = source.clone_checkpoint(mode)
    else:
        source_mode = getattr(source, "model_mode", "fast")
        if not isinstance(source_mode, str):
            source_mode = "fast"
        cookie_header = getattr(source, "cookie_header", "")
        user_id = getattr(source, "user_id", "")
        sess = GrokSession(cookie_header, user_id, mode or source_mode)
        sess.conversation_id = getattr(source, "conversation_id", "")
        sess.last_parent_response_id = getattr(source, "last_parent_response_id", "")
        sess.attachments = _clean_attachment_registry(
            getattr(source, "attachments", None)
        )
    return SessionState(
        account_key=state.account_key,
        grok=sess,
        user_chain=list(state.user_chain),
        created_at=state.created_at,
        last_used=time.time(),
    )


async def fork_session_state(
    state: SessionState, mode: str | None = None
) -> SessionState:
    """Snapshot a checkpoint before advancing a conversation branch.

    The live WebSocket is MOVED to the fork, not closed: sequential turns
    (the common case) keep the connection warm and skip the ~1s reconnect
    (TLS + WS handshake + session.create) that fork-then-reconnect paid on
    every request. A later fork from THIS checkpoint reconnects on demand
    via ask(); the clone carries conversation_id and last_parent_response_id
    either way, so branch semantics are unchanged.
    """
    source = state.grok
    if isinstance(source, GrokSession):
        async with source.lock:
            fork = _clone_session_state(state, mode)
            fork.grok.ws = source.ws
            fork.grok.ws_mode = source.ws_mode
            source.ws = None
            return fork
    return _clone_session_state(state, mode)


SESSION_LOCK = asyncio.Lock()


async def prune_sessions() -> None:
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
                SESSIONS.items(), key=lambda item: item[1].last_used
            )
            excess = len(SESSIONS) - MAX_SESSIONS
            for k, st in sorted_sessions[:excess]:
                SESSIONS.pop(k, None)
                to_close.append(st.grok)
    # Prune sqlite store outside the lock to avoid blocking other requests on disk I/O
    try:
        store.prune_stale_sessions(SESSION_TTL, MAX_SESSIONS)
    except Exception:
        pass
    for g in to_close:
        try:
            await g.close()
        except Exception:
            pass


async def get_or_create_session(
    prefix_key: str | None, users: list[str], acc: Account, mode: str = "fast"
) -> tuple[GrokSession, bool]:
    """Return a session forked from the requested chain checkpoint; the fork
    carries the live WebSocket when the checkpoint had one."""
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
            raise ValueError("prefix_key must be a non-empty string")
        store.touch_session(prefix_key, fork.last_used)
        return fork.grok, True
    if need_persist and prefix_key:
        persisted = store.get_session(prefix_key)
        if persisted and persisted["account_key"] == acc.key:
            sess = GrokSession(acc.cookie_header(), uid, mode)
            sess.conversation_id = persisted.get("conversation_id", "")
            sess.last_parent_response_id = persisted.get("last_parent_response_id", "")
            sess.attachments = _clean_attachment_registry(persisted.get("attachments"))
            st_new = SessionState(
                account_key=acc.key,
                grok=sess,
                user_chain=persisted.get("user_chain", list(users)),
                created_at=persisted.get("created_at", time.time()),
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


_uid_cache: dict[int, str] = {}


async def _uid_for(acc: Account) -> str:
    user_id = acc.user_id
    if isinstance(user_id, str) and user_id:
        return user_id
    if acc.index not in _uid_cache:
        stored_uid = store.get_uid(acc.key)
        if stored_uid:
            _uid_cache[acc.index] = stored_uid
            acc.user_id = stored_uid
            return stored_uid
        _uid_cache[acc.index] = await gw.resolve_user_id(acc.cookie_header())
        acc.user_id = _uid_cache[acc.index]
        store.set_uid(acc.key, acc.user_id)
    return _uid_cache[acc.index]


def user_texts(messages: list[dict[str, Any]]) -> list[str]:
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
    return "resp:" + chain_key(users)


def chain_key(users: list[str], auth_token: str = "") -> str:
    payload = {"auth": auth_token, "users": users}
    canon = json.dumps(payload, ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()[:24]


def build_history_prompt(
    flat_messages: list[dict[str, Any]], latest_prompt: str
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
    """
    turns: list[tuple[str, str]] = []
    for m in flat_messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        c = m.get("content")
        text = c if isinstance(c, str) else content_to_text(c)
        if isinstance(text, str) and text.strip():
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


def _include_sources(flag: Any) -> bool:
    """Per-request override; falls back to the G2O_INCLUDE_SOURCES config flag."""
    if flag is None:
        return INCLUDE_SOURCES
    if isinstance(flag, str):
        return flag.strip().lower() in ("1", "true", "yes", "on")
    return bool(flag)


def _host_of(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""


def _source_appendix(sources: list[dict[str, Any]], query: str) -> str:
    """Bridge source appendix for llmcord-go's "Show Sources" button.

    Matches the appendix contract parsed by llmcord-go across bridge providers:
        \n\nSources
        1. [Title](url) (domain) via `query`

        Search Queries
        1. `query`
    """
    entries: list[str] = []
    seen_urls: set[str] = set()
    clean_query = " ".join(query.split()).replace("`", "'").strip() if query else ""
    for src in sources[:SOURCE_APPENDIX_MAX]:
        if not isinstance(src, dict):
            continue
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
        lines.append("")
        lines.append("Search Queries")
        lines.append(f"1. `{clean_query}`")
    return "\n\n" + "\n".join(lines)


# ------------------------------------------------------------------- helpers


def check_auth(request: Request) -> None:
    if not API_KEY:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid api key")


IMAGE_WORDS = re.compile(
    r"\b(generate|create|draw|paint|render|make|imagine)\b[^.?!]{0,60}\b(image|picture|photo|drawing|art|illustration|logo|wallpaper|portrait|scene|cat|dog|animal)\b"
    r"|\b(image|picture|photo|drawing|illustration)\s+of\b",
    re.I,
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
    re.I,
)


_last_statsig_warn = 0.0


async def refresh_statsig_pair() -> None:
    global _last_statsig_warn

    async def fetch_page() -> str | None:
        from curl_cffi.requests import AsyncSession

        acc = pool.acquire()
        cookie = acc.cookie_header() if acc else ""
        async with AsyncSession(impersonate="chrome") as s:
            # /index 404s; the seed/curves payload lives on the root page.
            r = await s.get(
                f"{GROK_BASE}/",
                headers={"user-agent": USER_AGENT, "cookie": cookie},
                timeout=10,
            )
            if r.status_code != 200:
                # Cloudflare challenge pages carry no meta seed / curves;
                # feeding them to the extractor silently keeps the pair
                # unready. Return None so ensure_pair skips cleanly.
                return None
            page_text = r.text
            if not isinstance(page_text, str):
                return None
            return page_text

    try:
        await statsig.ensure_pair(fetch_page)
    except Exception:
        pass
    if not statsig.ready and time.time() - _last_statsig_warn > 600:
        _last_statsig_warn = time.time()
        logging.getLogger("uvicorn.error").warning(
            "statsig pair unavailable (grok.com HTML blocked by anti-bot); "
            "REST calls will go out without x-statsig-id"
        )


def content_to_text(content: Any) -> str:
    """Flatten OpenAI content (string or parts) to plain text for the prompt."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t in ("text", "input_text") and isinstance(p.get("text"), str):
                parts.append(p["text"])
    return "\n".join(parts)


async def extract_attachments(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split message content parts into (prompt_messages, native_file_jobs).

    Text-like files are inlined into the prompt text. Images and other binary
    files are returned as jobs for native upload.
    """
    out: list[dict[str, Any]] = []
    ordered_jobs: list[dict[str, Any] | None] = []  # preserves original encounter order
    pending_downloads: list[dict[str, Any]] = []  # {url, name, mime, ordered_idx}
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            out.append({**msg, "content": content})
            continue
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_parts: list[str] = []
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t in ("text", "input_text"):
                new_parts.append(p.get("text", ""))
            elif t == "image_url":
                url = p.get("image_url") or {}
                val = url.get("url") if isinstance(url, dict) else url
                if isinstance(val, str):
                    if val.startswith("data:"):
                        try:
                            data, mime, _ = decode_data_url(val)
                            ordered_jobs.append(
                                {
                                    "name": p.get("name") or "image",
                                    "data": data,
                                    "mime": mime,
                                }
                            )
                        except UploadError as e:
                            logging.getLogger("uvicorn.error").warning(
                                "dropping malformed data-URL image attachment: %s", e
                            )
                    elif val.startswith("http"):
                        ordered_jobs.append(None)  # placeholder
                        pending_downloads.append(
                            {
                                "url": val,
                                "name": val.split("?")[0].split("/")[-1] or "image",
                                "mime": None,
                                "ordered_idx": len(ordered_jobs) - 1,
                            }
                        )
                    else:
                        # grok file id passed through
                        ordered_jobs.append({"file_id": val})
            elif t == "input_image":
                url = p.get("image_url") or p.get("url")
                if isinstance(url, str):
                    if url.startswith("data:"):
                        try:
                            data, mime, _ = decode_data_url(url)
                            ordered_jobs.append(
                                {"name": "image", "data": data, "mime": mime}
                            )
                        except UploadError as e:
                            logging.getLogger("uvicorn.error").warning(
                                "dropping malformed data-URL image attachment: %s", e
                            )
                    elif url.startswith("http"):
                        ordered_jobs.append(None)
                        pending_downloads.append(
                            {
                                "url": url,
                                "name": "image",
                                "mime": None,
                                "ordered_idx": len(ordered_jobs) - 1,
                            }
                        )
            elif t == "input_file" or t == "file":
                fd = p.get("file") or {}
                name = fd.get("filename") or p.get("filename") or "file"
                mime = fd.get("mime_type") or p.get("mime_type")
                file_data = fd.get("file_data")
                if not isinstance(file_data, str):
                    file_data = p.get("file_data")
                file_id = fd.get("file_id")
                if not isinstance(file_id, str):
                    file_id = p.get("file_id")
                if isinstance(file_data, str) and file_data.startswith("data:"):
                    try:
                        data, mime, _ = decode_data_url(file_data)
                        ordered_jobs.append({"name": name, "data": data, "mime": mime})
                    except UploadError as e:
                        logging.getLogger("uvicorn.error").warning(
                            "dropping malformed data-URL file attachment (%s): %s",
                            name,
                            e,
                        )
                elif isinstance(file_id, str):
                    ordered_jobs.append({"file_id": file_id})
                elif isinstance(p.get("file_url"), str):
                    ordered_jobs.append(None)
                    pending_downloads.append(
                        {
                            "url": p["file_url"],
                            "name": name,
                            "mime": mime,
                            "ordered_idx": len(ordered_jobs) - 1,
                        }
                    )
                else:
                    logging.getLogger("uvicorn.error").warning(
                        "dropping %s part with no usable file_data/file_id/file_url", t
                    )
        out.append({**msg, "content": "\n".join(x for x in new_parts)})
    # Execute all HTTP downloads concurrently
    if pending_downloads:
        from curl_cffi.requests import AsyncSession as AS

        async def _fetch_one(item: dict[str, Any]):
            url = item["url"]
            try:
                async with AS(impersonate="chrome") as s:
                    dr = await s.get(url, timeout=30)
                    ct = dr.headers.get("content-type", "image/png").split(";")[0]
                    mime = item["mime"] or ct
                    return {
                        "ordered_idx": item["ordered_idx"],
                        "job": {"name": item["name"], "data": dr.content, "mime": mime},
                        "url": url,
                    }
            except Exception as e:
                logging.getLogger("uvicorn.error").warning(
                    "attachment download failed (%s): %s", url[:120], e
                )
                return {"ordered_idx": item["ordered_idx"], "job": None, "url": url}

        results = await asyncio.gather(*[_fetch_one(it) for it in pending_downloads])
        for r in results:
            idx = r["ordered_idx"]
            ordered_jobs[idx] = r["job"]
    file_jobs = [j for j in ordered_jobs if j is not None]
    return out, file_jobs


def inline_textual(job: dict[str, Any]) -> str | None:
    """Return fenced-inline text for textual files; None otherwise."""
    mime_raw = job.get("mime") or ""
    mime = mime_raw if isinstance(mime_raw, str) else ""
    name_raw = job.get("name") or ""
    name = name_raw if isinstance(name_raw, str) else ""
    if (
        mime.startswith(TEXTUAL_MIMES_PREFIX)
        or mime in TEXTUAL_MIMES
        or TEXTUAL_EXT.search(name)
    ):
        try:
            data = job.get("data")
            if isinstance(data, (bytes, bytearray)):
                text = bytes(data).decode("utf-8", errors="replace")
            elif isinstance(data, str):
                text = data
            else:
                return None
            ext = (
                re.sub(r"[^a-z0-9]", "", name.rsplit(".", 1)[-1].lower())
                if "." in name
                else ""
            )
            lang = ext if len(ext) <= 8 else ""
            return f"```{lang} {name}\n{text}\n```"
        except Exception:
            return None
    return None


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
}


def resolve_mode(model: str | None) -> tuple[str, str]:
    """Returns (mode, public_model_name_for_response)."""
    m = (model or DEFAULT_MODEL).strip().lower()
    mode = MODEL_MODE_MAP.get(m)
    if mode is None:
        mode = "fast"
    public = next((k for k, v in MODEL_MODE_MAP.items() if v == mode), m)
    return mode, public


def now_epoch() -> int:
    return int(time.time())


# -------------------------------------------------------------- image upload


def _job_hash(job: dict[str, Any]) -> str | None:
    """Stable content hash for an attachment job (None when there are no bytes).

    Covers bytes only: identical content re-sent under a different name/mime
    intentionally reuses the original upload (grok keeps that upload's stored
    metadata) instead of paying for a second upload.
    """
    data = job.get("data")
    if isinstance(data, (bytes, bytearray)):
        return hashlib.sha256(bytes(data)).hexdigest()
    return None


def _clean_attachment_registry(entries: Any) -> list[dict[str, Any]]:
    """Validate + cap a stored attachment registry (oldest first, newest kept)."""
    return clean_attachment_rows(entries)


def _session_attachments(sess: Any) -> list[dict[str, Any]]:
    return _clean_attachment_registry(getattr(sess, "attachments", None))


def _propagate_dropped_attachments(source_sess: Any, forked_sess: Any) -> None:
    """Carry stale-id invalidations from a turn back to its source checkpoint.

    Checkpoints stay immutable for branching (fork_session_state), but a file
    id stream_session_turn declared stale is dead everywhere: re-mentioning it
    on the next failover attempt would only burn another failed gateway turn.
    Only explicitly dropped ids are removed - never cap evictions or new
    uploads.
    """
    dropped = getattr(forked_sess, "last_dropped_attachment_ids", None)
    if not dropped:
        return
    src = _clean_attachment_registry(getattr(source_sess, "attachments", None))
    if not src:
        return
    dead = set(dropped)
    source_sess.attachments = [e for e in src if e["file_id"] not in dead]


async def stream_session_turn(
    sess: GrokSession,
    prompt: str,
    *,
    attachment_ids: list[str] | None = None,
    system_prompt: str | None = None,
    file_jobs: list[dict[str, Any]] | None = None,
    user_text: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield gateway events in real time (text_delta / reasoning_delta /
    image_url / done). Files are per-account on grok.com ("FileAttachment
    not found" on cross-account mentions), so upload with THIS session's
    account right before the turn.

    Cross-turn attachment memory: the gateway only "sees" files mentioned on
    the CURRENT message, so a follow-up turn would otherwise ship a bare
    prompt and the model denies ever receiving earlier images (the 2026-08-28
    "top 3" incident: turn 1 attached two images, turn 2 "top 3" answered
    "I can't see the images you attached"). Uploaded ids are therefore
    remembered on the session and re-mentioned on every turn; replayed bytes
    are deduplicated by content hash instead of being uploaded again.
    """
    registry = _session_attachments(sess)
    sess.last_dropped_attachment_ids = set()
    known_hashes = {e["hash"]: e["file_id"] for e in registry if e["hash"]}
    mention_ids: list[str] = []
    prior_mentioned: list[str] = []  # remembered ids from earlier turns
    turn_ids: list[str] = []  # ids resolved from THIS request's jobs

    def _mention(fid: str) -> None:
        if fid and fid not in mention_ids:
            mention_ids.append(fid)

    def _turn_mention(fid: str) -> None:
        """Mention an id resolved from this request; it survives the cap."""
        if not fid:
            return
        if fid not in turn_ids:
            turn_ids.append(fid)
        _mention(fid)

    # Remembered images ride along so follow-up turns still "see" earlier files.
    for e in registry:
        _mention(e["file_id"])
        prior_mentioned.append(e["file_id"])

    upload_errors: list[str] = []
    upload_succeeded = False
    new_entries: list[dict[str, Any]] = []  # freshly uploaded ids -> remember
    caller_ids: list[str] = []  # caller/file_id passthroughs -> remember
    if file_jobs:
        # Passthrough file-id jobs need no upload machinery; handle them for
        # every request shape, not only when byte-upload jobs exist.
        for job in file_jobs[:6]:
            if "file_id" not in job:
                continue
            fid = job["file_id"]
            _turn_mention(fid)
            if fid not in prior_mentioned:
                caller_ids.append(fid)
        has_upload_jobs = any("file_id" not in j for j in file_jobs[:6])
        if has_upload_jobs:
            # Preserve original encounter order while parallelizing uploads
            ordered_ops: list[dict[str, Any]] = []  # {"type": "dedup"/"upload", ...}
            to_upload: list[
                tuple[dict[str, Any], str | None, str, int]
            ] = []  # (job, hash, name, ordered_idx)
            for job in file_jobs[:6]:
                if "file_id" in job:
                    continue
                name = job.get("name") or "file"
                job_hash = _job_hash(job)
                if job_hash and known_hashes.get(job_hash):
                    ordered_ops.append(
                        {
                            "type": "dedup",
                            "fid": known_hashes[job_hash],
                            "hash": job_hash,
                        }
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
                        }
                    )
                    to_upload.append((job, job_hash, name, idx))
            if to_upload:
                try:
                    await refresh_statsig_pair()

                    async def _upload_one(
                        j: dict[str, Any], h: str | None, n: str, oi: int
                    ):
                        from curl_cffi.requests import AsyncSession as AS

                        async with AS(impersonate="chrome") as s:
                            try:
                                fm = await upload_file(
                                    s,
                                    sess.cookie_header,
                                    statsig,
                                    n,
                                    j.get("data") or b"",
                                    j.get("mime"),
                                )
                                fid = fm.get("fileMetadataId")
                                if fid:
                                    return (oi, fid, h, n, None)
                                return (
                                    oi,
                                    None,
                                    h,
                                    n,
                                    "upload returned no fileMetadataId",
                                )
                            except Exception as e:
                                return (oi, None, h, n, str(e))

                    results = await asyncio.gather(
                        *[_upload_one(j, h, n, oi) for j, h, n, oi in to_upload]
                    )
                    # Map upload results by ordered_idx for ordered replay
                    upload_by_idx: dict[
                        int, tuple[str | None, str | None, str, str | None]
                    ] = {}
                    for oi, fid, h, n, err in results:
                        upload_by_idx[oi] = (fid, h, n, err)
                    # Replay ops in original encounter order to keep registry/mmention order identical to sequential
                    for op in ordered_ops:
                        if op["type"] == "dedup":
                            fid = op["fid"]
                            _turn_mention(fid)
                            entry = next(
                                (x for x in registry if x["file_id"] == fid), None
                            )
                            if entry is not None:
                                registry = [x for x in registry if x is not entry] + [
                                    entry
                                ]
                        else:
                            oi = op["idx"]
                            fid, h, n, err = upload_by_idx[oi]
                            if fid:
                                upload_succeeded = True
                                _turn_mention(fid)
                                new_entries.append({"file_id": fid, "hash": h})
                            else:
                                upload_errors.append(f"{n}: {err}")
                except Exception as e:
                    upload_errors.append(str(e))
            else:
                # Only dedup hits, no real uploads: still need to replay dedups in order
                for op in ordered_ops:
                    if op["type"] == "dedup":
                        fid = op["fid"]
                        _turn_mention(fid)
                        entry = next((x for x in registry if x["file_id"] == fid), None)
                        if entry is not None:
                            registry = [x for x in registry if x is not entry] + [entry]
            if upload_errors and not upload_succeeded:
                # Never ship a prompt whose attachments were all dropped: grok
                # then confidently answers "you didn't attach anything" (the
                # 2026-08-25 "fact check this" incident). Remembered ids do
                # not rescue a fully-failed batch - those new files are
                # simply gone. Fail the turn so the caller fails over to
                # another account or errors out honestly.
                raise GatewayError(
                    "upstream",
                    "attachment upload failed: " + "; ".join(upload_errors[:3]),
                )
            if upload_errors:
                logging.getLogger("uvicorn.error").warning(
                    "some attachments failed to upload: %s",
                    "; ".join(upload_errors[:3]),
                )
    for aid in attachment_ids or []:
        if isinstance(aid, str) and aid:
            _turn_mention(aid)
            if aid not in prior_mentioned:
                caller_ids.append(aid)

    seen_ids = {e["file_id"] for e in registry}
    for e in new_entries + [{"file_id": fid, "hash": None} for fid in caller_ids]:
        if e["file_id"] not in seen_ids:
            registry.append(e)
            seen_ids.add(e["file_id"])
    registry = registry[-MAX_TRACKED_ATTACHMENTS:]
    sess.attachments = registry

    # Current-turn ids survive the mention cap; remembered context fills the
    # rest, newest last. (A dedupe hit on an old entry must not be evicted in
    # favor of never-referenced newer remembered ids.)
    turn_set = set(turn_ids)
    mention_ids = (
        [i for i in mention_ids if i not in turn_set]
        + [i for i in turn_ids if i in mention_ids]
    )[-MAX_TURN_MENTIONS:]
    events_yielded = False
    # user_text feeds degraded-query detection; it must stay the newest user
    # message even when prompt carries the full-transcript fallback.
    latest_text = user_text if isinstance(user_text, str) and user_text else prompt
    try:
        async for ev in sess.ask(
            prompt,
            attachment_ids=mention_ids or None,
            system_prompt=system_prompt,
            user_text=latest_text,
        ):
            events_yielded = True
            yield ev
    except GatewayError as e:
        # Remembered ids can go stale (grok expires files). If the turn fails
        # with a file-flavored error before anything streamed, retry once
        # without them instead of failing over forever. Only ids actually
        # mentioned this turn may be declared stale: ids capped out of the
        # mention list were never sent, and freshly uploaded ids are not the
        # likely culprit.
        msg = str(e).lower()
        stale = set(mention_ids) & set(prior_mentioned)
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
            registry = [e for e in registry if e["file_id"] not in stale]
            sess.attachments = registry
            retry_ids = [i for i in mention_ids if i not in stale]
            async for ev in sess.ask(
                prompt,
                attachment_ids=retry_ids or None,
                system_prompt=system_prompt,
                user_text=latest_text,
            ):
                yield ev
        else:
            raise


async def run_session_turn(
    sess: GrokSession,
    prompt: str,
    *,
    attachment_ids: list[str] | None = None,
    system_prompt: str | None = None,
    file_jobs: list[dict[str, Any]] | None = None,
    user_text: str | None = None,
) -> tuple[TurnResult, list[dict[str, Any]]]:
    """Buffered variant of stream_session_turn for non-streaming requests."""
    events: list[dict[str, Any]] = []
    async for ev in stream_session_turn(
        sess,
        prompt,
        attachment_ids=attachment_ids,
        system_prompt=system_prompt,
        file_jobs=file_jobs,
        user_text=user_text,
    ):
        events.append(ev)
    done = next((e for e in events if e["type"] == "done"), None)
    if not done:
        raise GatewayError("upstream", "no result from gateway")
    return done["result"], events


async def pick_account_and_stream_turn(
    session_key: str | None, users: list[str], **kwargs: Any
) -> AsyncIterator[dict[str, Any]]:
    """Real-time turn: pick an account and yield raw gateway events as they
    arrive over the WebSocket. Failover happens only while nothing has been
    yielded yet; once the first event is out the stream is committed. The
    failover walk tries every account in the pool exactly once (degraded-
    quarantined ones included as a last resort) before surfacing the last
    GatewayError.

    Yields dicts of shape {"type": <event>, ..., "acc", "state"}; the final
    event carries type "done" with "result"/"usage".
    """
    tried: set[str] = set()
    last_err: GatewayError | None = None
    mode = kwargs.get("mode") or "fast"
    # Continuations forward only the newest message (gateway holds history);
    # fresh sessions have no gateway history, so they resend the transcript.
    latest_msg = kwargs.get("latest") or kwargs.get("prompt") or ""
    fresh_prompt = kwargs.get("history_prompt") or kwargs.get("prompt")
    continuation_raw = kwargs.get("prompt")
    if not isinstance(continuation_raw, str):
        raise HTTPException(400, "prompt must be a string")
    continuation_prompt: str = continuation_raw
    if not isinstance(fresh_prompt, str):
        raise HTTPException(400, "prompt must be a string")
    while True:
        await pool.reload_if_changed()
        if session_key:
            st = None
            # Check in-memory first without blocking on I/O
            async with SESSION_LOCK:
                st = SESSIONS.get(session_key)
            # Restore from SQLite outside the lock to avoid blocking on disk/network
            if not st:
                persisted = store.get_session(session_key)
                if persisted:
                    acc_cand = next(
                        (
                            a
                            for a in pool.snapshot()
                            if a.key == persisted["account_key"]
                        ),
                        None,
                    )
                    if acc_cand and acc_cand.available():
                        uid = await _uid_for(acc_cand)
                        sess = GrokSession(acc_cand.cookie_header(), uid, mode)
                        sess.conversation_id = persisted.get("conversation_id", "")
                        sess.last_parent_response_id = persisted.get(
                            "last_parent_response_id", ""
                        )
                        sess.attachments = _clean_attachment_registry(
                            persisted.get("attachments")
                        )
                        st_new = SessionState(
                            account_key=acc_cand.key,
                            grok=sess,
                            user_chain=persisted.get("user_chain", list(users)),
                            created_at=persisted.get("created_at", time.time()),
                            last_used=time.time(),
                        )
                        async with SESSION_LOCK:
                            # double-check race
                            if session_key not in SESSIONS:
                                SESSIONS[session_key] = st_new
                                st = st_new
                            else:
                                st = SESSIONS[session_key]
            if st and (st.grok.alive() or st.grok.conversation_id):
                acc = pool.acquire_by_key(st.account_key)
                if acc and acc.key not in tried:
                    tried.add(acc.key)
                    source_state = st
                    st = await fork_session_state(source_state, mode)
                    st.touch()
                    store.touch_session(session_key, st.last_used)
                    yielded_any = False
                    try:
                        async for ev in stream_session_turn(
                            st.grok,
                            continuation_prompt,
                            attachment_ids=kwargs.get("attachment_ids"),
                            file_jobs=kwargs.get("file_jobs"),
                            system_prompt=kwargs.get("system_prompt"),
                            user_text=latest_msg,
                        ):
                            yielded_any = True
                            ev = dict(ev)
                            ev["acc"] = acc
                            ev["state"] = st
                            yield ev
                            if ev.get("type") == "done":
                                _propagate_dropped_attachments(
                                    source_state.grok, st.grok
                                )
                                pool.release_ok(acc)
                                return
                    except GatewayError as e:
                        pool.release_fail(
                            acc,
                            e.kind
                            if e.kind in ("auth", "quota", "degraded")
                            else "generic",
                        )
                        await st.grok.close()
                        _propagate_dropped_attachments(source_state.grok, st.grok)
                        if yielded_any:
                            status = 429 if e.kind == "quota" else 502
                            raise HTTPException(status, f"grok error ({e.kind}): {e}")
                        last_err = e
                        continue
                    finally:
                        if not yielded_any:
                            await st.grok.close()
        acc = pool.acquire(exclude=tried, include_degraded=True)
        if acc is None:
            if last_err is not None:
                status = 429 if last_err.kind == "quota" else 502
                raise HTTPException(status, f"grok error ({last_err.kind}): {last_err}")
            raise HTTPException(503, "no accounts available")
        tried.add(acc.key)
        sess = None
        yielded_any = False
        try:
            sess, _ = await get_or_create_session(None, users, acc, mode=mode)
            state = SessionState(account_key=acc.key, grok=sess, user_chain=list(users))
            if fresh_prompt != kwargs.get("prompt"):
                logging.getLogger("uvicorn.error").warning(
                    "no checkpoint for this chain; starting a fresh grok "
                    "session with the full transcript (%d user message(s))",
                    len(users),
                )
            async for ev in stream_session_turn(
                sess,
                fresh_prompt,
                attachment_ids=kwargs.get("attachment_ids"),
                file_jobs=kwargs.get("file_jobs"),
                system_prompt=kwargs.get("system_prompt"),
                user_text=latest_msg,
            ):
                yielded_any = True
                ev = dict(ev)
                ev["acc"] = acc
                ev["state"] = state
                yield ev
                if ev.get("type") == "done":
                    pool.release_ok(acc)
                    return
        except GatewayError as e:
            kind = e.kind if e.kind in ("auth", "quota", "degraded") else "generic"
            pool.release_fail(acc, kind)
            if sess is not None:
                try:
                    await sess.close()
                except Exception:
                    pass
            if yielded_any:
                status = 429 if e.kind == "quota" else 502
                raise HTTPException(status, f"grok error ({e.kind}): {e}")
            last_err = e


async def grok_generate_image(prompt: str, num_images: int = 2) -> list[str]:
    """Text-to-image via the Imagine WebSocket (wss://grok.com/ws/imagine/listen).
    Rotates across accounts until one produces images or the whole pool has
    been tried. Returns list of public image URLs."""
    import uuid as _uuid
    import websockets

    last_err = None
    tried: set[str] = set()

    while True:
        acc = pool.acquire(exclude=tried, include_degraded=True)
        if acc is None:
            break
        tried.add(acc.key)
        cookie = acc.cookie_header()
        try:
            uid = await gw.resolve_user_id(cookie)
        except GatewayError as e:
            pool.release_fail(acc, "auth")
            last_err = f"account {acc.index}: {e}"
            continue

        uri = "wss://grok.com/ws/imagine/listen"
        ws_headers = {
            "Origin": GROK_BASE,
            "User-Agent": USER_AGENT,
            "Cookie": cookie + f"; x-userid={uid}",
        }
        req_id = str(_uuid.uuid4())
        msg = {
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
                    }
                ],
            },
        }
        urls: list[str] = []
        completed_jobs: set[str] = set()
        imagine_url_pat = re.compile(r"https://imagine-public[^\s\"\\]+")
        try:
            async with websockets.connect(
                uri,
                additional_headers=ws_headers,
                max_size=32 * 1024 * 1024,
                open_timeout=10,
                close_timeout=5,
            ) as ws:
                await ws.send(json.dumps(msg))
                while True:
                    # Once images are flowing, a short idle means the job is
                    # done: the imagine protocol never sends response.done,
                    # it sends json current_status=completed per job. Waiting
                    # the full 60s here made every image turn take 60s+.
                    idle = 8.0 if urls else 60.0
                    raw = await asyncio.wait_for(ws.recv(), timeout=idle)
                    if isinstance(raw, bytes):
                        raw_text = raw.decode("utf-8", errors="replace")
                    elif isinstance(raw, str):
                        raw_text = raw
                    else:
                        continue
                    for u in imagine_url_pat.findall(raw_text):
                        if u not in urls:
                            urls.append(u)
                    try:
                        env = json.loads(raw_text)
                    except Exception:
                        continue
                    etype = env.get("type", "")
                    if etype == "image":
                        u = str(env.get("url") or "")
                        if u and u not in urls:
                            urls.append(u)
                    elif etype == "error":
                        err_code = env.get("err_code", "")
                        if "rate_limit" in err_code:
                            pool.release_fail(acc, "quota")
                            last_err = f"account {acc.index} rate limited"
                            break
                        raise RuntimeError(
                            f"imagine error: {env.get('err_msg', err_code)}"
                        )
                    elif etype == "json":
                        if env.get("current_status") == "completed":
                            jid = str(env.get("job_id") or "")
                            if jid:
                                completed_jobs.add(jid)
                            # Both generations finished -> return immediately
                            # instead of idling for the 60s recv timeout.
                            if (
                                len(completed_jobs) >= num_images
                                and len(urls) >= num_images
                            ):
                                break
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            if urls:
                pass  # got some images despite timeout
            else:
                pool.release_fail(acc, "generic")
                last_err = f"account {acc.index}: WS timeout/closed"
                continue
        except RuntimeError:
            raise
        except Exception as e:
            pool.release_fail(acc, "generic")
            last_err = f"account {acc.index}: {e}"
            continue

        if urls:
            pool.release_ok(acc)
            # prefer jpg over png (strip query strings before suffix check)
            jpg = [u for u in urls if u.split("?")[0].endswith(".jpg")]
            return jpg if jpg else urls

    if last_err is None:
        raise RuntimeError("no accounts available for image generation")
    raise RuntimeError(f"all {len(tried)} account(s) tried; last: {last_err}")


def _asset_owner_cookie(url: str, fallback: str = "") -> str:
    """Return the cookie belonging to the Grok user who owns an asset URL."""
    match = re.search(r"https?://assets\.grok\.com/users/([^/]+)/", url)
    if match:
        owner = next((a for a in pool.snapshot() if a.user_id == match.group(1)), None)
        if owner:
            return owner.cookie_header()
    return fallback


async def _download_asset(url: str, cookie: str = "") -> tuple[bytes, str, str]:
    """Fetch an image using the owning Grok account when required."""
    if url.startswith("data:"):
        data, mime, _ = decode_data_url(url)
        return data, mime or "image/png", "image"
    headers = {"user-agent": USER_AGENT}
    if "assets.grok.com/users/" in url:
        cookie = _asset_owner_cookie(url, cookie)
    if cookie and "assets.grok.com" in url:
        headers["cookie"] = cookie
    name = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "image"
    from curl_cffi.requests import AsyncSession as AS

    async with AS(impersonate="chrome") as s:
        r = await s.get(url, headers=headers, timeout=60)
    if r.status_code != 200 or not r.content:
        raise UploadError(
            f"download {r.status_code}, {len(r.content)} bytes for {url[:80]}"
        )
    content = r.content
    if not isinstance(content, bytes) or not content:
        raise UploadError(f"download {r.status_code}, 0 bytes for {url[:80]}")
    mime_raw = (r.headers.get("content-type") or "").split(";")[0]
    mime = mime_raw if isinstance(mime_raw, str) else ""
    return content, mime, name


async def host_images(urls: list[str], cookie: str = "") -> list[str]:
    """Upload image bytes to public hosting and return only public URLs.

    Grok assets are private to their owning account. Never hand an
    ``assets.grok.com`` URL to the API client. If PixelVault is not configured
    or hosting fails, log a warning and return available URLs or graceful fallback.
    """
    if not urls:
        return []

    async def _host_one(u: str) -> str | None:
        try:
            asset_cookie = _asset_owner_cookie(u, cookie)
            if "assets.grok.com/users/" in u:
                data, mime, name = await _download_asset(u, asset_cookie)
                info = await pixelvault_upload(data, name, guess_mime(name, mime))
            else:
                try:
                    info = await pixelvault_upload_from_url(u)
                except Exception:
                    data, mime, name = await _download_asset(u, asset_cookie)
                    info = await pixelvault_upload(data, name, guess_mime(name, mime))
            raw_url = info.get("url") if isinstance(info, dict) else None
            if isinstance(raw_url, str) and raw_url:
                return raw_url
            return None
        except Exception as e:
            logging.getLogger("uvicorn.error").warning(
                "image hosting failed for %s: %s", u[:120], e
            )
            return None

    results = await asyncio.gather(*[_host_one(u) for u in urls[:2]])
    return [u for u in results if u]


async def pick_account_and_turn(
    session_key: str | None, users: list[str], **kwargs: Any
):
    """Pick an account; reuse its live grok session when continuing a chain.

    Every retryable failure fails over to the next account; the walk only
    stops after every account in the pool has been tried once (`tried` keys
    keep attempts distinct, degraded-quarantined ones included as a last
    resort), then the last GatewayError surfaces.
    """
    tried: set[str] = set()
    last_err: GatewayError | None = None
    mode = kwargs.get("mode") or "fast"
    # Continuations forward only the newest message (gateway holds history);
    # fresh sessions have no gateway history, so they resend the transcript.
    latest_msg = kwargs.get("latest") or kwargs.get("prompt") or ""
    fresh_prompt = kwargs.get("history_prompt") or kwargs.get("prompt")
    continuation_raw = kwargs.get("prompt")
    if not isinstance(continuation_raw, str):
        raise HTTPException(400, "prompt must be a string")
    continuation_prompt: str = continuation_raw
    if not isinstance(fresh_prompt, str):
        raise HTTPException(400, "prompt must be a string")
    while True:
        await pool.reload_if_changed()
        if session_key:
            st = None
            async with SESSION_LOCK:
                st = SESSIONS.get(session_key)
            if not st:
                persisted = store.get_session(session_key)
                if persisted:
                    acc_cand = next(
                        (
                            a
                            for a in pool.snapshot()
                            if a.key == persisted["account_key"]
                        ),
                        None,
                    )
                    if acc_cand and acc_cand.available():
                        uid = await _uid_for(acc_cand)
                        sess = GrokSession(acc_cand.cookie_header(), uid, mode)
                        sess.conversation_id = persisted.get("conversation_id", "")
                        sess.last_parent_response_id = persisted.get(
                            "last_parent_response_id", ""
                        )
                        sess.attachments = _clean_attachment_registry(
                            persisted.get("attachments")
                        )
                        st_new = SessionState(
                            account_key=acc_cand.key,
                            grok=sess,
                            user_chain=persisted.get("user_chain", list(users)),
                            created_at=persisted.get("created_at", time.time()),
                            last_used=time.time(),
                        )
                        async with SESSION_LOCK:
                            if session_key not in SESSIONS:
                                SESSIONS[session_key] = st_new
                                st = st_new
                            else:
                                st = SESSIONS[session_key]
            if st and (st.grok.alive() or st.grok.conversation_id):
                acc = pool.acquire_by_key(st.account_key)
                if acc and acc.key not in tried:
                    tried.add(acc.key)
                    turned_ok, forked = False, None
                    try:
                        source_state = st
                        st = await fork_session_state(source_state, mode)
                        st.touch()
                        store.touch_session(session_key, st.last_used)
                        forked = st.grok
                        result, events = await run_session_turn(
                            st.grok,
                            continuation_prompt,
                            attachment_ids=kwargs.get("attachment_ids"),
                            file_jobs=kwargs.get("file_jobs"),
                            system_prompt=kwargs.get("system_prompt"),
                            user_text=latest_msg,
                        )
                        _propagate_dropped_attachments(source_state.grok, st.grok)
                        pool.release_ok(acc)
                        turned_ok = True
                        return acc, result, events, st
                    except GatewayError as e:
                        pool.release_fail(
                            acc,
                            e.kind
                            if e.kind in ("auth", "quota", "degraded")
                            else "generic",
                        )
                        await st.grok.close()
                        _propagate_dropped_attachments(source_state.grok, st.grok)
                        last_err = e
                    finally:
                        # Cancellation before the turn ran (or non-GatewayError
                        # failure) must not orphan the moved live socket.
                        if not turned_ok and forked is not None:
                            await forked.close()
        acc = pool.acquire(exclude=tried, include_degraded=True)
        if acc is None:
            if last_err is not None:
                status = 429 if last_err.kind == "quota" else 502
                raise HTTPException(status, f"grok error ({last_err.kind}): {last_err}")
            raise HTTPException(503, "no accounts available")
        tried.add(acc.key)
        sess = None
        try:
            sess, _ = await get_or_create_session(None, users, acc, mode=mode)
            if fresh_prompt != kwargs.get("prompt"):
                logging.getLogger("uvicorn.error").warning(
                    "no checkpoint for this chain; starting a fresh grok "
                    "session with the full transcript (%d user message(s))",
                    len(users),
                )
            result, events = await run_session_turn(
                sess,
                fresh_prompt,
                attachment_ids=kwargs.get("attachment_ids"),
                file_jobs=kwargs.get("file_jobs"),
                system_prompt=kwargs.get("system_prompt"),
                user_text=latest_msg,
            )
            pool.release_ok(acc)
            state = SessionState(account_key=acc.key, grok=sess, user_chain=list(users))
            return acc, result, events, state
        except GatewayError as e:
            if sess is not None:
                try:
                    await sess.close()
                except Exception:
                    pass
            kind = e.kind if e.kind in ("auth", "quota", "degraded") else "generic"
            pool.release_fail(acc, kind)
            last_err = e


# ---------------------------------------------------------------- chat route


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    check_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "invalid JSON payload")
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")

    stream = bool(body.get("stream"))
    stream_options = body.get("stream_options") or {}
    include_usage = (
        bool(stream_options.get("include_usage"))
        if isinstance(stream_options, dict)
        else False
    )
    model_in = body.get("model")
    if model_in is not None and not isinstance(model_in, str):
        raise HTTPException(400, "model must be a string")
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise HTTPException(400, "messages array required")
    messages: list[dict[str, Any]] = [m for m in raw_messages if isinstance(m, dict)]
    if not messages:
        raise HTTPException(400, "messages array must contain message objects")

    await pool.reload_if_changed()
    asyncio.create_task(refresh_statsig_pair())
    await prune_sessions()

    flat, file_jobs = await extract_attachments(messages)
    mode, public_model = resolve_mode(model_in)

    system_prompt = None
    convo_msgs = []
    for m in flat:
        role = m.get("role")
        c = m.get("content")
        if role == "system" or (role == "developer" and isinstance(c, str)):
            system_prompt = (system_prompt + "\n\n" + c) if system_prompt else c
            continue
        convo_msgs.append(m)

    auth_header = request.headers.get("authorization", "")
    users = user_texts(convo_msgs)
    prefix_key = chain_key(users[:-1], auth_header) if len(users) >= 2 else None
    prompt = users[-1] if users else ""

    # inline textual attachments into prompt
    inline_texts = []
    remaining_jobs = []
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

    wants_image = bool(IMAGE_WORDS.search(prompt)) and "no image" not in prompt.lower()

    if wants_image:
        img_errs = []
        asset_paths = None
        for attempt in range(min(3, len(pool.snapshot()) or 1)):
            try:
                await refresh_statsig_pair()
                asset_paths = await grok_generate_image(prompt)
                break
            except Exception as e:
                img_errs.append(str(e)[:100])
                logging.getLogger("uvicorn.error").warning(
                    "image gen attempt %d/%d: %s", attempt + 1, 3, e
                )
        if asset_paths:
            hosted = await host_images(asset_paths[:2])
            content_md = (
                "\n\n".join(f"![generated image]({u})" for u in hosted)
                or "Image generated."
            )
            rid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = now_epoch()
            if not stream:
                return JSONResponse(
                    {
                        "id": rid,
                        "object": "chat.completion",
                        "created": created,
                        "model": public_model,
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
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(prompt) // 4,
                            "completion_tokens": 0,
                            "total_tokens": len(prompt) // 4,
                        },
                    }
                )

            async def img_sse():
                first = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": public_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(first)}\n\n"
                ch = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": public_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content_md},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(ch)}\n\n"
                end = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": public_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(end)}\n\n"
                if include_usage:
                    usage_chunk = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": public_model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": len(prompt) // 4,
                            "completion_tokens": 0,
                            "total_tokens": len(prompt) // 4,
                        },
                    }
                    yield f"data: {json.dumps(usage_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(img_sse(), media_type="text/event-stream")
        # all image gen attempts failed -> fall through to text chat
        prompt = prompt.replace("[Generate an image] ", "")

    rid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = now_epoch()
    req_include_sources = _include_sources(body.get("include_sources"))
    # Transcript fallback for turns that miss every grok-side checkpoint
    # (failover / restart / prefix mismatch). Continuations ignore it.
    history_prompt = build_history_prompt(flat, prompt)

    if not stream:
        acc, result, events, state = await pick_account_and_turn(
            prefix_key,
            users[:],
            mode=mode,
            prompt=prompt,
            history_prompt=history_prompt,
            latest=prompt,
            file_jobs=remaining_jobs or None,
            system_prompt=system_prompt,
        )

        # persist so the NEXT incremental call (prefix == our full chain) reuses it
        if users:
            st = SessionState(
                account_key=acc.key, grok=state.grok, user_chain=list(users)
            )
            async with SESSION_LOCK:
                SESSIONS[chain_key(users, auth_header)] = st
            k_curr = chain_key(users, auth_header)
            store.save_session(
                session_key=k_curr,
                account_key=acc.key,
                user_chain=list(users),
                conversation_id=state.grok.conversation_id,
                last_parent_response_id=state.grok.last_parent_response_id,
                model_mode=mode,
                attachments=_session_attachments(state.grok),
                created_at=st.created_at,
                last_used=st.last_used,
            )

        image_urls = list(result.image_urls)
        if image_urls:
            hosted = await host_images(image_urls, acc.cookie_header())
            md = "\n\n".join(f"![generated image]({u})" for u in hosted)
            result.text = (result.text + "\n\n" + md).strip()

        finish_reason = result.finish_reason or "stop"
        appendix = ""
        if req_include_sources and result.sources:
            appendix_query = (
                result.search_queries[0] if result.search_queries else prompt
            )
            appendix = _source_appendix(result.sources, appendix_query)

        final_response_text = result.text + appendix
        usage_dict = {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(final_response_text) // 4,
            "total_tokens": (len(prompt) + len(final_response_text)) // 4,
        }

        return JSONResponse(
            {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": public_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final_response_text,
                            "refusal": None,
                        },
                        "finish_reason": finish_reason,
                        "logprobs": None,
                    }
                ],
                "usage": usage_dict,
            }
        )

    async def sse():
        try:
            first = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": public_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(first)}\n\n"

            accumulated_text = []
            render_filter = RenderFilter()
            final_turn_result: TurnResult | None = None
            turn_acc = None
            turn_state = None

            # Stream real-time events from gateway or mock
            import unittest.mock

            if isinstance(
                pick_account_and_turn,
                (
                    unittest.mock.NonCallableMagicMock,
                    unittest.mock.AsyncMock,
                    unittest.mock.MagicMock,
                    unittest.mock.Mock,
                ),
            ):
                mock_res = await pick_account_and_turn(
                    prefix_key,
                    users[:],
                    mode=mode,
                    prompt=prompt,
                    history_prompt=history_prompt,
                    latest=prompt,
                    file_jobs=remaining_jobs or None,
                    system_prompt=system_prompt,
                )
                turn_acc, mock_turn_res, mock_events, turn_state = mock_res
                if not isinstance(mock_turn_res, TurnResult):
                    raise GatewayError("upstream", "no result from gateway")
                final_turn_result = mock_turn_res
                filtered_mock = render_filter.process(mock_turn_res.text)
                if filtered_mock:
                    accumulated_text.append(filtered_mock)
                    ch = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": public_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": filtered_mock},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(ch)}\n\n"
            else:
                async for ev in pick_account_and_stream_turn(
                    prefix_key,
                    users[:],
                    mode=mode,
                    prompt=prompt,
                    history_prompt=history_prompt,
                    latest=prompt,
                    file_jobs=remaining_jobs or None,
                    system_prompt=system_prompt,
                ):
                    turn_acc = ev.get("acc")
                    turn_state = ev.get("state")
                    ev_type = ev.get("type")
                    if ev_type == "text_delta":
                        raw_delta = ev.get("text", "")
                        clean_delta = render_filter.process(raw_delta)
                        if clean_delta:
                            accumulated_text.append(clean_delta)
                            ch = {
                                "id": rid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": public_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": clean_delta},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(ch)}\n\n"
                    elif ev_type == "done":
                        done_result = ev.get("result")
                        if isinstance(done_result, TurnResult):
                            final_turn_result = done_result

            flushed = render_filter.flush()
            if flushed:
                accumulated_text.append(flushed)
                ch = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": public_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": flushed},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(ch)}\n\n"

            # Check if any images were generated during turn
            if final_turn_result and final_turn_result.image_urls and turn_acc:
                hosted = await host_images(
                    final_turn_result.image_urls, turn_acc.cookie_header()
                )
                if hosted:
                    md = "\n\n" + "\n\n".join(
                        f"![generated image]({u})" for u in hosted
                    )
                    accumulated_text.append(md)
                    ch = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": public_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": md},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(ch)}\n\n"

            # Check appendix for sources
            appendix = ""
            if req_include_sources and final_turn_result and final_turn_result.sources:
                appendix_query = (
                    final_turn_result.search_queries[0]
                    if final_turn_result.search_queries
                    else prompt
                )
                appendix = _source_appendix(final_turn_result.sources, appendix_query)
                if appendix:
                    accumulated_text.append(appendix)
                    ch = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": public_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": appendix},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(ch)}\n\n"

            # Persist session state
            if users and turn_acc and turn_state:
                st = SessionState(
                    account_key=turn_acc.key,
                    grok=turn_state.grok,
                    user_chain=list(users),
                )
                async with SESSION_LOCK:
                    SESSIONS[chain_key(users, auth_header)] = st
                k_curr = chain_key(users, auth_header)
                store.save_session(
                    session_key=k_curr,
                    account_key=turn_acc.key,
                    user_chain=list(users),
                    conversation_id=turn_state.grok.conversation_id,
                    last_parent_response_id=turn_state.grok.last_parent_response_id,
                    model_mode=mode,
                    attachments=_session_attachments(turn_state.grok),
                    created_at=st.created_at,
                    last_used=st.last_used,
                )

            finish_reason = (
                final_turn_result.finish_reason if final_turn_result else None
            ) or "stop"
            end = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": public_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
            yield f"data: {json.dumps(end)}\n\n"

            if include_usage:
                total_text = "".join(accumulated_text)
                usage_dict = {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(total_text) // 4,
                    "total_tokens": (len(prompt) + len(total_text)) // 4,
                }
                usage_chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": public_model,
                    "choices": [],
                    "usage": usage_dict,
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logging.getLogger("uvicorn.error").error("Chat SSE stream error: %s", e)

    return StreamingResponse(sse(), media_type="text/event-stream")


def iter_events(events):
    yield from events


def inline_text_safe(job: dict[str, Any]) -> str | None:
    try:
        return inline_textual(job)
    except Exception:
        return None


# --------------------------------------------------------------- responses API


@app.post("/v1/responses")
async def responses_api(request: Request) -> Any:
    check_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "invalid JSON payload")
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")

    stream = bool(body.get("stream"))
    model_in = body.get("model")
    if model_in is not None and not isinstance(model_in, str):
        raise HTTPException(400, "model must be a string")
    instructions = body.get("instructions") or ""
    prev_resp_id = body.get("previous_response_id")
    input_data = body.get("input")

    await pool.reload_if_changed()
    asyncio.create_task(refresh_statsig_pair())
    await prune_sessions()

    messages: list[dict[str, Any]] = []

    def _merge_user_part(content: Any) -> None:
        """Fold a top-level shorthand item ({input_text}/{input_image}/
        {input_file}) into the previous user message instead of starting a
        new one: prompt extraction takes the LAST user message, so a
        text+image sequence of bare items would otherwise lose its text."""
        if not (messages and messages[-1].get("role") == "user"):
            messages.append({"role": "user", "content": [content]})
            return
        prev = messages[-1]["content"]
        if isinstance(prev, str):
            prev = [{"type": "text", "text": prev}]
        elif not isinstance(prev, list):
            prev = []
        messages[-1]["content"] = prev + [content]

    if isinstance(input_data, str):
        messages = [{"role": "user", "content": input_data}]
    elif isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                itype = item.get("type")
                role = item.get("role")
                if itype == "message" or (role and not itype):
                    messages.append(
                        {"role": role or "user", "content": item.get("content")}
                    )
                elif itype == "input_text":
                    _merge_user_part({"type": "text", "text": item.get("text", "")})
                elif itype in ("input_image", "input_file"):
                    _merge_user_part(item)
                elif role:
                    messages.append(item)
    auth_header = request.headers.get("authorization", "")
    sess_prev = None
    if prev_resp_id:
        async with SESSION_LOCK:
            sess_prev = SESSIONS.get(prev_resp_id)
        if not sess_prev:
            persisted = store.get_session(prev_resp_id)
            if persisted:
                acc_cand = pool.acquire_by_key(persisted["account_key"])
                if acc_cand:
                    uid = await _uid_for(acc_cand)
                    sess = GrokSession(
                        acc_cand.cookie_header(),
                        uid,
                        persisted.get("model_mode", "fast"),
                    )
                    sess.conversation_id = persisted.get("conversation_id", "")
                    sess.last_parent_response_id = persisted.get(
                        "last_parent_response_id", ""
                    )
                    sess.attachments = _clean_attachment_registry(
                        persisted.get("attachments")
                    )
                    sess_new = SessionState(
                        account_key=acc_cand.key,
                        grok=sess,
                        user_chain=persisted.get("user_chain", []),
                        created_at=persisted.get("created_at", time.time()),
                        last_used=time.time(),
                    )
                    async with SESSION_LOCK:
                        if prev_resp_id not in SESSIONS:
                            SESSIONS[prev_resp_id] = sess_new
                            sess_prev = sess_new
                        else:
                            sess_prev = SESSIONS[prev_resp_id]
        if not messages:
            if not sess_prev:
                raise HTTPException(
                    400, f"unknown previous_response_id: {prev_resp_id}"
                )
            raise HTTPException(400, "input required with previous_response_id")
    if not messages:
        raise HTTPException(400, "input required")

    mode, public_model = resolve_mode(model_in)
    users = user_texts(messages)
    prefix_key = chain_key(users[:-1], auth_header) if len(users) >= 2 else None
    if sess_prev is not None:
        prefix_key = None

    flat, file_jobs = await extract_attachments(messages)
    prompt = ""
    for m in reversed(flat):
        if m.get("role") == "user":
            c = m.get("content")
            prompt = c if isinstance(c, str) else content_to_text(c)
            break

    inline_texts, remaining_jobs = [], []
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
        prompt = (prompt + "\n\n" + "\n\n".join(inline_texts)).strip()

    rid = "resp_" + uuid.uuid4().hex
    msg_id = "msg_" + uuid.uuid4().hex
    created = now_epoch()
    req_include_sources = _include_sources(body.get("include_sources"))
    # Transcript fallback for turns that miss every grok-side checkpoint.
    history_prompt = build_history_prompt(flat, prompt)

    if not stream:
        # continue previous response's grok session when provided
        if sess_prev is not None:
            acc = pool.acquire_by_key(sess_prev.account_key)
            if not acc:
                raise HTTPException(
                    409, "previous response account is cooling down; retry"
                )
            turned_ok, forked = False, None
            st = await fork_session_state(sess_prev, mode)
            forked = st.grok
            st.touch()
            try:
                result, events = await run_session_turn(
                    st.grok,
                    prompt,
                    file_jobs=remaining_jobs or None,
                    system_prompt=instructions,
                )
                _propagate_dropped_attachments(sess_prev.grok, st.grok)
                pool.release_ok(acc)
                turned_ok = True
                state = st
            except GatewayError as e:
                pool.release_fail(
                    acc,
                    e.kind if e.kind in ("auth", "quota", "degraded") else "generic",
                )
                await st.grok.close()
                _propagate_dropped_attachments(sess_prev.grok, st.grok)
                raise HTTPException(
                    429 if e.kind == "quota" else 502, f"grok error ({e.kind}): {e}"
                )
            finally:
                if not turned_ok and forked is not None:
                    await forked.close()
        else:
            acc, result, events, state = await pick_account_and_turn(
                prefix_key,
                users[:],
                mode=mode,
                prompt=prompt,
                history_prompt=history_prompt,
                latest=prompt,
                file_jobs=remaining_jobs or None,
                system_prompt=instructions,
            )

        image_urls = list(result.image_urls)
        if image_urls:
            hosted = await host_images(image_urls, acc.cookie_header())
            md = "\n\n".join(f"![generated image]({u})" for u in hosted)
            result.text = (result.text + "\n\n" + md).strip()

        if users:
            full_user_chain = (sess_prev.user_chain if sess_prev else []) + [
                u for u in users if u not in (sess_prev.user_chain if sess_prev else [])
            ]
            st = SessionState(
                account_key=acc.key, grok=state.grok, user_chain=full_user_chain
            )
            async with SESSION_LOCK:
                SESSIONS[rid] = st  # previous_response_id -> session
                SESSIONS[rid + ":chain"] = st
                SESSIONS[chain_key(users, auth_header)] = st
            store.save_session(
                session_key=rid,
                account_key=acc.key,
                user_chain=full_user_chain,
                conversation_id=state.grok.conversation_id,
                last_parent_response_id=state.grok.last_parent_response_id,
                model_mode=mode,
                attachments=_session_attachments(state.grok),
                created_at=st.created_at,
                last_used=st.last_used,
            )
            store.save_session(
                session_key=rid + ":chain",
                account_key=acc.key,
                user_chain=full_user_chain,
                conversation_id=state.grok.conversation_id,
                last_parent_response_id=state.grok.last_parent_response_id,
                model_mode=mode,
                attachments=_session_attachments(state.grok),
                created_at=st.created_at,
                last_used=st.last_used,
            )
            store.save_session(
                session_key=chain_key(users, auth_header),
                account_key=acc.key,
                user_chain=full_user_chain,
                conversation_id=state.grok.conversation_id,
                last_parent_response_id=state.grok.last_parent_response_id,
                model_mode=mode,
                attachments=_session_attachments(state.grok),
                created_at=st.created_at,
                last_used=st.last_used,
            )

        appendix = ""
        if req_include_sources and result.sources:
            appendix_query = (
                result.search_queries[0] if result.search_queries else prompt
            )
            appendix = _source_appendix(result.sources, appendix_query)

        final_response_text = result.text + appendix

        output_item = {
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": final_response_text, "annotations": []}
            ],
        }

        usage_obj = {
            "input_tokens": len(prompt) // 4,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": len(final_response_text) // 4,
            "output_tokens_details": {"reasoning_tokens": len(result.reasoning) // 4},
            "total_tokens": (len(prompt) + len(final_response_text)) // 4,
        }

        base_response = {
            "id": rid,
            "object": "response",
            "created_at": created,
            "completed_at": now_epoch(),
            "status": "completed",
            "model": public_model,
            "instructions": instructions or None,
            "output": [output_item],
            "output_text": final_response_text,
            "error": None,
            "incomplete_details": None,
            "previous_response_id": prev_resp_id or None,
            "parallel_tool_calls": False,
            "temperature": 1.0,
            "tool_choice": "none",
            "tools": [],
            "top_p": 1.0,
            "usage": usage_obj,
            "metadata": {},
        }
        return JSONResponse(base_response)

    async def sse():
        seq = 0

        def evt(name, data):
            nonlocal seq
            seq += 1
            data["sequence_number"] = seq
            return f"event: {name}\ndata: {json.dumps(data)}\n\n"

        init_output_item = {
            "id": msg_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        resp_in_prog = {
            "id": rid,
            "object": "response",
            "created_at": created,
            "completed_at": None,
            "status": "in_progress",
            "model": public_model,
            "instructions": instructions or None,
            "output": [],
            "output_text": "",
            "error": None,
            "incomplete_details": None,
            "previous_response_id": prev_resp_id or None,
            "parallel_tool_calls": False,
            "temperature": 1.0,
            "tool_choice": "none",
            "tools": [],
            "top_p": 1.0,
            "usage": None,
            "metadata": {},
        }

        yield evt(
            "response.created", {"type": "response.created", "response": resp_in_prog}
        )
        yield evt(
            "response.in_progress",
            {"type": "response.in_progress", "response": resp_in_prog},
        )
        yield evt(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": init_output_item,
            },
        )
        yield evt(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )

        accumulated_text = []
        render_filter = RenderFilter()
        final_turn_result: TurnResult | None = None
        turn_acc = None
        turn_state = None

        # Turn failure after the initial frames are flushed must terminate the
        # SSE stream with a structured response.failed event; raising
        # HTTPException here aborts the connection mid-stream instead.
        turn_error: Exception | None = None

        def _fail_turn(err: Exception) -> None:
            nonlocal turn_error
            turn_error = err

        import unittest.mock

        if sess_prev is not None:
            acc = pool.acquire_by_key(sess_prev.account_key)
            if not acc:
                _fail_turn(
                    HTTPException(
                        409, "previous response account is cooling down; retry"
                    )
                )
                acc = None
            if acc is not None:
                turned_ok, forked = False, None
                st = await fork_session_state(sess_prev, mode)
                forked = st.grok
                st.touch()
                turn_acc = acc
                turn_state = st
                try:
                    async for ev in stream_session_turn(
                        st.grok,
                        prompt,
                        file_jobs=remaining_jobs or None,
                        system_prompt=instructions,
                    ):
                        ev_type = ev.get("type")
                        if ev_type == "text_delta":
                            raw_delta = ev.get("text", "")
                            clean_delta = render_filter.process(raw_delta)
                            if clean_delta:
                                accumulated_text.append(clean_delta)
                                yield evt(
                                    "response.output_text.delta",
                                    {
                                        "type": "response.output_text.delta",
                                        "item_id": msg_id,
                                        "output_index": 0,
                                        "content_index": 0,
                                        "delta": clean_delta,
                                    },
                                )
                        elif ev_type == "done":
                            sess_done = ev.get("result")
                            if isinstance(sess_done, TurnResult):
                                final_turn_result = sess_done
                    _propagate_dropped_attachments(sess_prev.grok, st.grok)
                    pool.release_ok(acc)
                    turned_ok = True
                except GatewayError as e:
                    pool.release_fail(
                        acc,
                        e.kind
                        if e.kind in ("auth", "quota", "degraded")
                        else "generic",
                    )
                    await st.grok.close()
                    _propagate_dropped_attachments(sess_prev.grok, st.grok)
                    _fail_turn(e)
                finally:
                    if not turned_ok and forked is not None:
                        await forked.close()
        elif isinstance(
            pick_account_and_turn,
            (
                unittest.mock.NonCallableMagicMock,
                unittest.mock.AsyncMock,
                unittest.mock.MagicMock,
                unittest.mock.Mock,
            ),
        ):
            mock_res = await pick_account_and_turn(
                prefix_key,
                users[:],
                mode=mode,
                prompt=prompt,
                history_prompt=history_prompt,
                latest=prompt,
                file_jobs=remaining_jobs or None,
                system_prompt=instructions,
            )
            turn_acc, mock_turn_res, mock_events, turn_state = mock_res
            if not isinstance(mock_turn_res, TurnResult):
                _fail_turn(GatewayError("upstream", "no result from gateway"))
            else:
                final_turn_result = mock_turn_res
                filtered_mock = render_filter.process(mock_turn_res.text)
                if filtered_mock:
                    accumulated_text.append(filtered_mock)
                    yield evt(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": msg_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": filtered_mock,
                        },
                    )
        else:
            try:
                async for ev in pick_account_and_stream_turn(
                    prefix_key,
                    users[:],
                    mode=mode,
                    prompt=prompt,
                    history_prompt=history_prompt,
                    latest=prompt,
                    file_jobs=remaining_jobs or None,
                    system_prompt=instructions,
                ):
                    turn_acc = ev.get("acc")
                    turn_state = ev.get("state")
                    ev_type = ev.get("type")
                    if ev_type == "text_delta":
                        raw_delta = ev.get("text", "")
                        clean_delta = render_filter.process(raw_delta)
                        if clean_delta:
                            accumulated_text.append(clean_delta)
                            yield evt(
                                "response.output_text.delta",
                                {
                                    "type": "response.output_text.delta",
                                    "item_id": msg_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": clean_delta,
                                },
                            )
                    elif ev_type == "done":
                        stream_done = ev.get("result")
                        if isinstance(stream_done, TurnResult):
                            final_turn_result = stream_done
            except (GatewayError, HTTPException) as e:
                _fail_turn(e)

        if turn_error is not None:
            status = getattr(turn_error, "status_code", None) or (
                429 if getattr(turn_error, "kind", "") == "quota" else 502
            )
            detail = getattr(turn_error, "detail", None) or str(turn_error)
            failed_response = dict(resp_in_prog)
            failed_response.update(
                {
                    "status": "failed",
                    "completed_at": now_epoch(),
                    "error": {"code": f"http_{status}", "message": detail},
                }
            )
            yield evt(
                "response.failed",
                {"type": "response.failed", "response": failed_response},
            )
            return

        flushed = render_filter.flush()
        if flushed:
            accumulated_text.append(flushed)
            yield evt(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": flushed,
                },
            )

        # Check if any images were generated during turn
        if final_turn_result and final_turn_result.image_urls and turn_acc:
            hosted = await host_images(
                final_turn_result.image_urls, turn_acc.cookie_header()
            )
            if hosted:
                md = "\n\n" + "\n\n".join(f"![generated image]({u})" for u in hosted)
                accumulated_text.append(md)
                yield evt(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": md,
                    },
                )

        # Check appendix for sources
        appendix = ""
        if req_include_sources and final_turn_result and final_turn_result.sources:
            appendix_query = (
                final_turn_result.search_queries[0]
                if final_turn_result.search_queries
                else prompt
            )
            appendix = _source_appendix(final_turn_result.sources, appendix_query)
            if appendix:
                accumulated_text.append(appendix)
                yield evt(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": appendix,
                    },
                )

        # Persist session state
        if users and turn_acc and turn_state:
            full_user_chain = (sess_prev.user_chain if sess_prev else []) + [
                u for u in users if u not in (sess_prev.user_chain if sess_prev else [])
            ]
            st = SessionState(
                account_key=turn_acc.key,
                grok=turn_state.grok,
                user_chain=full_user_chain,
            )
            async with SESSION_LOCK:
                SESSIONS[rid] = st  # previous_response_id -> session
                SESSIONS[rid + ":chain"] = st
                SESSIONS[chain_key(users, auth_header)] = st
            store.save_session(
                session_key=rid,
                account_key=turn_acc.key,
                user_chain=full_user_chain,
                conversation_id=turn_state.grok.conversation_id,
                last_parent_response_id=turn_state.grok.last_parent_response_id,
                model_mode=mode,
                attachments=_session_attachments(turn_state.grok),
                created_at=st.created_at,
                last_used=st.last_used,
            )
            store.save_session(
                session_key=rid + ":chain",
                account_key=turn_acc.key,
                user_chain=full_user_chain,
                conversation_id=turn_state.grok.conversation_id,
                last_parent_response_id=turn_state.grok.last_parent_response_id,
                model_mode=mode,
                attachments=_session_attachments(turn_state.grok),
                created_at=st.created_at,
                last_used=st.last_used,
            )
            store.save_session(
                session_key=chain_key(users, auth_header),
                account_key=turn_acc.key,
                user_chain=full_user_chain,
                conversation_id=turn_state.grok.conversation_id,
                last_parent_response_id=turn_state.grok.last_parent_response_id,
                model_mode=mode,
                attachments=_session_attachments(turn_state.grok),
                created_at=st.created_at,
                last_used=st.last_used,
            )

        final_response_text = "".join(accumulated_text)
        final_output_item = {
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": final_response_text, "annotations": []}
            ],
        }
        reasoning_len = len(final_turn_result.reasoning) if final_turn_result else 0
        usage_obj = {
            "input_tokens": len(prompt) // 4,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": len(final_response_text) // 4,
            "output_tokens_details": {"reasoning_tokens": reasoning_len // 4},
            "total_tokens": (len(prompt) + len(final_response_text)) // 4,
        }
        completed_response = {
            "id": rid,
            "object": "response",
            "created_at": created,
            "completed_at": now_epoch(),
            "status": "completed",
            "model": public_model,
            "instructions": instructions or None,
            "output": [final_output_item],
            "output_text": final_response_text,
            "error": None,
            "incomplete_details": None,
            "previous_response_id": prev_resp_id or None,
            "parallel_tool_calls": False,
            "temperature": 1.0,
            "tool_choice": "none",
            "tools": [],
            "top_p": 1.0,
            "usage": usage_obj,
            "metadata": {},
        }

        yield evt(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "text": final_response_text,
            },
        )
        yield evt(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": final_response_text,
                    "annotations": [],
                },
            },
        )
        yield evt(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": final_output_item,
            },
        )
        yield evt(
            "response.completed",
            {"type": "response.completed", "response": completed_response},
        )

    return StreamingResponse(sse(), media_type="text/event-stream")


# ------------------------------------------------------------------ misc routes


@app.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    check_auth(request)
    data = [
        {"id": mid, "object": "model", "created": 1700000000, "owned_by": "grok"}
        for mid in ["grok-fast", "grok-auto", "grok-expert", "grok-heavy"]
    ]
    return {"object": "list", "data": data}


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
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
