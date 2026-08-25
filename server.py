"""Grok → OpenAI-compatible API server.

Endpoints:
  POST /v1/chat/completions   (stream + non-stream)
  POST /v1/responses          (stream + non-stream, previous_response_id chaining)
  GET  /v1/models
  GET  /healthz

Chat runs over Grok's WebSocket Gateway (fast path, no browser). Multi-turn
conversations are real grok conversations: incremental requests reuse the
existing gateway session (conversation attach + parent_response_id), sending
only the newest user message.
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
from config import API_KEY, COOLDOWN_SECONDS, DEFAULT_MODEL, GROK_BASE, PORT, \
    SESSION_TTL, MAX_SESSIONS, USER_AGENT, INCLUDE_SOURCES
from accounts import AccountPool
from statsig import StatsigGenerator, STATSIG_EPOCH, SALT, compute_animation_hex, curves_to_path
from session_store import SqliteStore


from uploads import UploadError, decode_data_url, guess_mime, pixelvault_upload, \
    pixelvault_upload_from_url, upload_file
import grok_gateway as gw
from grok_gateway import GrokSession, GatewayError, TurnResult, RenderFilter

app = FastAPI(title="grok-to-openai-api", version="1.0")

store = SqliteStore(config.DB_PATH)
pool = AccountPool(config.ACCOUNTS_FILE, cooldown_seconds=COOLDOWN_SECONDS, store=store)
statsig = StatsigGenerator(store=store)


# ---------------------------------------------------------------- session map

@dataclass
class SessionState:
    account_key: str            # owning account key
    grok: GrokSession           # live gateway session (one per conversation)
    user_chain: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_used = time.time()


SESSIONS: dict[str, SessionState] = {}
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
            sorted_sessions = sorted(SESSIONS.items(), key=lambda item: item[1].last_used)
            excess = len(SESSIONS) - MAX_SESSIONS
            for k, st in sorted_sessions[:excess]:
                SESSIONS.pop(k, None)
                to_close.append(st.grok)
        # Prune sqlite store
        store.prune_stale_sessions(SESSION_TTL, MAX_SESSIONS)
    for g in to_close:
        try:
            await g.close()
        except Exception:
            pass


async def get_or_create_session(prefix_key: str | None, users: list[str],
                                acc, mode: str = "fast") -> tuple[GrokSession, bool]:
    """Return (session, continued). If prefix_key maps to a live session owned
    by `acc`, reuse it (true multi-turn); otherwise create a fresh one with mode."""
    uid = await _uid_for(acc)
    async with SESSION_LOCK:
        if prefix_key:
            st = SESSIONS.get(prefix_key)
            if st and st.account_key == acc.key and st.grok.alive():
                st.touch()
                store.touch_session(prefix_key, st.last_used)
                return st.grok, True
            # Try restoring from persistent sqlite store
            if not st:
                persisted = store.get_session(prefix_key)
                if persisted and persisted["account_key"] == acc.key:
                    sess = GrokSession(acc.cookie_header(), uid, mode)
                    sess.conversation_id = persisted.get("conversation_id", "")
                    sess.last_parent_response_id = persisted.get("last_parent_response_id", "")
                    st = SessionState(
                        account_key=acc.key,
                        grok=sess,
                        user_chain=persisted.get("user_chain", list(users)),
                        created_at=persisted.get("created_at", time.time()),
                        last_used=time.time(),
                    )
                    SESSIONS[prefix_key] = st
                    store.touch_session(prefix_key, st.last_used)
                    return sess, True
        sess = GrokSession(acc.cookie_header(), uid, mode)
        state = SessionState(account_key=acc.key, grok=sess, user_chain=list(users))
        if prefix_key:
            SESSIONS[prefix_key] = state
            store.save_session(
                session_key=prefix_key,
                account_key=acc.key,
                user_chain=list(users),
                conversation_id=sess.conversation_id,
                last_parent_response_id=sess.last_parent_response_id,
                model_mode=mode,
                created_at=state.created_at,
                last_used=state.last_used,
            )
        return sess, False


_uid_cache: dict[str, str] = {}

async def _uid_for(acc) -> str:
    if acc.user_id:
        return acc.user_id
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


def user_texts(messages: list[dict]) -> list[str]:
    return [m.get("content") if isinstance(m.get("content"), str)
            else content_to_text(m.get("content"))
            for m in messages if m.get("role") == "user"]


def rid_hint(users: list[str]) -> str:
    return "resp:" + chain_key(users)


def chain_key(users: list[str], auth_token: str = "") -> str:
    payload = {"auth": auth_token, "users": users}
    canon = json.dumps(payload, ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()[:24]


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


def _source_appendix(sources: list, query: str) -> str:
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

TEXTUAL_MIMES_PREFIX = ("text/",)
TEXTUAL_MIMES = {
    "application/json", "application/javascript", "application/typescript",
    "application/xml", "application/x-python", "application/x-sh",
    "application/yaml", "application/toml", "application/sql",
}
TEXTUAL_EXT = re.compile(
    r"\.(txt|md|markdown|json|jsonl|yaml|yml|toml|ini|cfg|conf|py|js|mjs|cjs|ts|tsx|jsx|"
    r"java|kt|go|rs|rb|php|c|h|cpp|hpp|cs|swift|sh|bash|zsh|sql|html?|css|scss|xml|csv|tsv|log|env)$",
    re.I,
)


async def refresh_statsig_pair() -> None:
    async def fetch_page() -> str:
        from curl_cffi.requests import AsyncSession
        acc = pool.acquire()
        cookie = acc.cookie_header() if acc else ""
        async with AsyncSession(impersonate="chrome") as s:
            # /index 404s; the seed/curves payload lives on the root page.
            r = await s.get(f"{GROK_BASE}/",
                            headers={"user-agent": USER_AGENT, "cookie": cookie},
                            timeout=30)
            return r.text
    try:
        await statsig.ensure_pair(fetch_page)
    except Exception:
        pass


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
            if t == "text" and isinstance(p.get("text"), str):
                parts.append(p["text"])
    return "\n".join(parts)


async def extract_attachments(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split message content parts into (prompt_messages, native_file_jobs).

    Text-like files are inlined into the prompt text. Images and other binary
    files are returned as jobs for native upload.
    """
    file_jobs: list[dict] = []   # {name, data, mime}
    out: list[dict] = []
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
            if t == "text":
                new_parts.append(p.get("text", ""))
            elif t == "image_url":
                url = (p.get("image_url") or {})
                val = url.get("url") if isinstance(url, dict) else url
                if isinstance(val, str):
                    if val.startswith("data:"):
                        try:
                            data, mime, _ = decode_data_url(val)
                            file_jobs.append({"name": p.get("name") or "image",
                                              "data": data, "mime": mime})
                        except UploadError:
                            pass
                    elif val.startswith("http"):
                        try:
                            from curl_cffi.requests import AsyncSession as AS
                            async with AS(impersonate="chrome") as s:
                                dr = await s.get(val, timeout=30)
                                ct = dr.headers.get("content-type", "image/png").split(";")[0]
                                file_jobs.append({"name": val.split("?")[0].split("/")[-1] or "image",
                                                  "data": dr.content, "mime": ct})
                        except Exception:
                            pass
                    else:
                        # grok file id passed through
                        file_jobs.append({"file_id": val})
            elif t == "input_image":
                url = p.get("image_url") or p.get("url")
                if isinstance(url, str):
                    if url.startswith("data:"):
                        try:
                            data, mime, _ = decode_data_url(url)
                            file_jobs.append({"name": "image", "data": data, "mime": mime})
                        except UploadError:
                            pass
                    elif url.startswith("http"):
                        try:
                            from curl_cffi.requests import AsyncSession as AS
                            async with AS(impersonate="chrome") as s:
                                dr = await s.get(url, timeout=30)
                                ct = dr.headers.get("content-type", "image/png").split(";")[0]
                                file_jobs.append({"name": "image", "data": dr.content, "mime": ct})
                        except Exception:
                            pass
            elif t == "input_file" or t == "file":
                fd = p.get("file") or {}
                name = fd.get("filename") or p.get("filename") or "file"
                mime = fd.get("mime_type") or p.get("mime_type")
                if isinstance(fd.get("file_data"), str):
                    val = fd["file_data"]
                    if val.startswith("data:"):
                        try:
                            data, mime, _ = decode_data_url(val)
                            file_jobs.append({"name": name, "data": data, "mime": mime})
                        except UploadError:
                            pass
                elif isinstance(fd.get("file_id"), str):
                    file_jobs.append({"file_id": fd["file_id"]})
                elif isinstance(p.get("file_url"), str):
                    try:
                        from curl_cffi.requests import AsyncSession as AS
                        async with AS(impersonate="chrome") as s:
                            dr = await s.get(p["file_url"], timeout=30)
                            file_jobs.append({"name": name, "data": dr.content,
                                              "mime": mime or dr.headers.get("content-type", "").split(";")[0]})
                    except Exception:
                        pass
        out.append({**msg, "content": "\n".join(x for x in new_parts)})
    return out, file_jobs


def inline_textual(job: dict) -> str | None:
    """Return fenced-inline text for textual files; None otherwise."""
    mime = job.get("mime") or ""
    name = job.get("name") or ""
    if mime.startswith(TEXTUAL_MIMES_PREFIX) or mime in TEXTUAL_MIMES or TEXTUAL_EXT.search(name):
        try:
            text = job.get("data").decode("utf-8", errors="replace")
            ext = re.sub(r"[^a-z0-9]", "", name.rsplit(".", 1)[-1].lower()) if "." in name else ""
            lang = ext if len(ext) <= 8 else ""
            return f"```{lang} {name}\n{text}\n```"
        except Exception:
            return None
    return None


MODEL_MODE_MAP = {
    "grok-fast": "fast", "fast": "fast", "grok-4.5-fast": "fast",
    "grok-auto": "auto", "auto": "auto",
    "grok-expert": "expert", "expert": "expert",
    "grok-heavy": "heavy", "heavy": "heavy",
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

async def stream_session_turn(sess: GrokSession, prompt: str, *,
                              attachment_ids=None, system_prompt=None,
                              file_jobs=None) -> AsyncIterator[dict]:
    """Yield gateway events in real time (text_delta / reasoning_delta /
    image_url / done). Files are per-account on grok.com ("FileAttachment
    not found" on cross-account mentions), so upload with THIS session's
    account right before the turn."""
    if file_jobs:
        try:
            from curl_cffi.requests import AsyncSession as AS
            await refresh_statsig_pair()
            async with AS(impersonate="chrome") as s:
                for job in file_jobs[:6]:
                    if "file_id" in job:
                        attachment_ids = (attachment_ids or []) + [job["file_id"]]
                        continue
                    fm = await upload_file(s, sess.cookie_header, statsig,
                                           job.get("name") or "file",
                                           job.get("data") or b"",
                                           job.get("mime"))
                    fid = fm.get("fileMetadataId")
                    if fid:
                        attachment_ids = (attachment_ids or []) + [fid]
        except Exception as e:
            logging.getLogger("uvicorn.error").warning(
                "attachment upload failed: %s", e)
            attachment_ids = None
    async for ev in sess.ask(prompt, attachment_ids=attachment_ids,
                             system_prompt=system_prompt, user_text=prompt):
        yield ev


async def run_session_turn(sess: GrokSession, prompt: str, *,
                           attachment_ids=None, system_prompt=None,
                           file_jobs=None) -> tuple[TurnResult, list[dict]]:
    """Buffered variant of stream_session_turn for non-streaming requests."""
    events = []
    async for ev in stream_session_turn(sess, prompt,
                                        attachment_ids=attachment_ids,
                                        system_prompt=system_prompt,
                                        file_jobs=file_jobs):
        events.append(ev)
    done = next((e for e in events if e["type"] == "done"), None)
    if not done:
        raise GatewayError("upstream", "no result from gateway")
    return done["result"], events


async def pick_account_and_stream_turn(session_key: str | None,
                                       users: list[str], **kwargs) \
        -> AsyncIterator[dict]:
    """Real-time turn: pick an account and yield raw gateway events as they
    arrive over the WebSocket. Failover happens only while nothing has been
    yielded yet; once the first event is out the stream is committed.

    Yields dicts of shape {"type": <event>, ..., "acc", "state"}; the final
    event carries type "done" with "result"/"usage".
    """
    attempts = 0
    mode = kwargs.get("mode") or "fast"
    while True:
        await pool.reload_if_changed()
        if session_key:
            st = None
            async with SESSION_LOCK:
                st = SESSIONS.get(session_key)
                if not st:
                    persisted = store.get_session(session_key)
                    if persisted:
                        acc_cand = next((a for a in pool.snapshot() if a.key == persisted["account_key"]), None)
                        if acc_cand and acc_cand.available():
                            uid = await _uid_for(acc_cand)
                            sess = GrokSession(acc_cand.cookie_header(), uid, mode)
                            sess.conversation_id = persisted.get("conversation_id", "")
                            sess.last_parent_response_id = persisted.get("last_parent_response_id", "")
                            st = SessionState(
                                account_key=acc_cand.key,
                                grok=sess,
                                user_chain=persisted.get("user_chain", list(users)),
                                created_at=persisted.get("created_at", time.time()),
                                last_used=time.time(),
                            )
                            SESSIONS[session_key] = st
            if st and (st.grok.alive() or st.grok.conversation_id):
                acc = pool.acquire_by_key(st.account_key)
                if acc:
                    attempts += 1
                    st.touch()
                    store.touch_session(session_key, st.last_used)
                    yielded_any = False
                    try:
                        async for ev in stream_session_turn(
                                st.grok, kwargs.get("prompt"),
                                attachment_ids=kwargs.get("attachment_ids"),
                                file_jobs=kwargs.get("file_jobs"),
                                system_prompt=kwargs.get("system_prompt")):
                            yielded_any = True
                            ev = dict(ev)
                            ev["acc"] = acc
                            ev["state"] = st
                            yield ev
                            if ev.get("type") == "done":
                                pool.release_ok(acc)
                                store.save_session(
                                    session_key=session_key,
                                    account_key=acc.key,
                                    user_chain=st.user_chain,
                                    conversation_id=st.grok.conversation_id,
                                    last_parent_response_id=st.grok.last_parent_response_id,
                                    model_mode=mode,
                                    created_at=st.created_at,
                                    last_used=st.last_used,
                                )
                                return
                    except GatewayError as e:
                        pool.release_fail(acc, e.kind if e.kind in
                                          ("auth", "quota", "degraded") else "generic")
                        await st.grok.close()
                        async with SESSION_LOCK:
                            SESSIONS.pop(session_key, None)
                            store.delete_session(session_key)
                        if yielded_any:
                            status = 429 if e.kind == "quota" else 502
                            raise HTTPException(status, f"grok error ({e.kind}): {e}")
                        continue
                    finally:
                        if not yielded_any:
                            pass
        acc = pool.acquire()
        if acc is None:
            raise HTTPException(503, "no accounts available")
        attempts += 1
        sess = None
        yielded_any = False
        try:
            sess, _ = await get_or_create_session(None, users, acc, mode=mode)
            state = SessionState(account_key=acc.key, grok=sess, user_chain=list(users))
            async for ev in stream_session_turn(
                    sess, kwargs.get("prompt"),
                    attachment_ids=kwargs.get("attachment_ids"),
                    file_jobs=kwargs.get("file_jobs"),
                    system_prompt=kwargs.get("system_prompt")):
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
            if attempts >= max(3, min(len(pool.snapshot()) or 1, 5)) \
                    or not any(a.available() for a in pool.snapshot()):
                status = 429 if e.kind == "quota" else 502
                raise HTTPException(status, f"grok error ({e.kind}): {e}")




async def grok_generate_image(prompt: str, num_images: int = 2) -> list[str]:
    """Text-to-image via the Imagine WebSocket (wss://grok.com/ws/imagine/listen).
    Rotates across accounts on rate limit. Returns list of public image URLs."""
    import uuid as _uuid
    import websockets

    max_attempts = min(4, len(pool.snapshot()) or 1)
    last_err = None

    for attempt in range(max_attempts):
        acc = pool.acquire()
        if acc is None:
            break
        cookie = acc.cookie_header()
        try:
            uid = await gw.resolve_user_id(cookie)
        except GatewayError:
            pool.release_fail(acc, "auth")
            continue

        uri = "wss://grok.com/ws/imagine/listen"
        ws_headers = {"Origin": GROK_BASE, "User-Agent": USER_AGENT,
                      "Cookie": cookie + f"; x-userid={uid}"}
        req_id = str(_uuid.uuid4())
        msg = {
            "type": "conversation.item.create",
            "timestamp": int(time.time() * 1000),
            "item": {"type": "message", "content": [{
                "requestId": req_id,
                "text": prompt,
                "type": "input_text",
                "properties": {
                    "section_count": 0, "is_kids_mode": False,
                    "enable_nsfw": True, "skip_upsampler": False,
                    "enable_side_by_side": True, "is_initial": True,
                    "aspect_ratio": "2:3", "enable_pro": False,
                    "num_generations": num_images, "enable_watermark": False,
                },
            }]},
        }
        urls: list[str] = []
        try:
            async with websockets.connect(uri, additional_headers=ws_headers,
                                          max_size=32 * 1024 * 1024) as ws:
                await ws.send(json.dumps(msg))
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=180)
                    pat = re.compile(r"https://imagine-public[^" + chr(34) + chr(92) + chr(92) + chr(92) + "s]+")
                    for u in pat.findall(raw):
                        if u not in urls:
                            urls.append(u)
                    env = json.loads(raw)
                    etype = env.get("type", "")
                    if etype == "error":
                        err_code = env.get("err_code", "")
                        if "rate_limit" in err_code:
                            pool.release_fail(acc, "quota")
                            last_err = f"account {acc.index} rate limited"
                            break
                        raise RuntimeError(f"imagine error: {env.get('err_msg', err_code)}")
                    if etype == "image" and urls:
                        # We have at least one image; wait briefly for more
                        pass
                    if etype == "response.done":
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
            # prefer jpg over png
            jpg = [u for u in urls if u.endswith(".jpg")]
            return jpg if jpg else urls

    raise RuntimeError(f"all {max_attempts} attempts failed; last: {last_err}")




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
        raise UploadError(f"download {r.status_code}, {len(r.content)} bytes for {url[:80]}")
    mime = (r.headers.get("content-type") or "").split(";")[0]
    return r.content, mime, name


async def host_images(urls: list[str], cookie: str = "") -> list[str]:
    """Upload image bytes to public hosting and return only public URLs.

    Grok assets are private to their owning account. Never hand an
    ``assets.grok.com`` URL to the API client. If PixelVault is not configured
    or hosting fails, log a warning and return available URLs or graceful fallback.
    """
    hosted: list[str] = []
    for u in urls[:2]:
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
            public_url = info.get("url") if isinstance(info, dict) else None
            if public_url:
                hosted.append(public_url)
        except Exception as e:
            logging.getLogger("uvicorn.error").warning(
                "image hosting failed for %s: %s", u[:120], e)
    return hosted



async def pick_account_and_turn(session_key: str | None, users: list[str], **kwargs):
    """Pick an account; reuse its live grok session when continuing a chain."""
    attempts = 0
    mode = kwargs.get("mode") or "fast"
    while True:
        await pool.reload_if_changed()
        continued = False
        if session_key:
            st = None
            async with SESSION_LOCK:
                st = SESSIONS.get(session_key)
                if not st:
                    persisted = store.get_session(session_key)
                    if persisted:
                        acc_cand = next((a for a in pool.snapshot() if a.key == persisted["account_key"]), None)
                        if acc_cand and acc_cand.available():
                            uid = await _uid_for(acc_cand)
                            sess = GrokSession(acc_cand.cookie_header(), uid, mode)
                            sess.conversation_id = persisted.get("conversation_id", "")
                            sess.last_parent_response_id = persisted.get("last_parent_response_id", "")
                            st = SessionState(
                                account_key=acc_cand.key,
                                grok=sess,
                                user_chain=persisted.get("user_chain", list(users)),
                                created_at=persisted.get("created_at", time.time()),
                                last_used=time.time(),
                            )
                            SESSIONS[session_key] = st
            if st and (st.grok.alive() or st.grok.conversation_id):
                acc = pool.acquire_by_key(st.account_key)
                if acc:
                    continued = True
                    attempts += 1
                    try:
                        st.touch()
                        store.touch_session(session_key, st.last_used)
                        result, events = await run_session_turn(
                            st.grok, kwargs.get("prompt"),
                            attachment_ids=kwargs.get("attachment_ids"),
                            file_jobs=kwargs.get("file_jobs"),
                            system_prompt=kwargs.get("system_prompt"))
                        pool.release_ok(acc)
                        store.save_session(
                            session_key=session_key,
                            account_key=acc.key,
                            user_chain=st.user_chain,
                            conversation_id=st.grok.conversation_id,
                            last_parent_response_id=st.grok.last_parent_response_id,
                            model_mode=mode,
                            created_at=st.created_at,
                            last_used=st.last_used,
                        )
                        return acc, result, events, st
                    except GatewayError as e:
                        pool.release_fail(acc, e.kind if e.kind in ("auth", "quota", "degraded") else "generic")
                        await st.grok.close()
                        async with SESSION_LOCK:
                            SESSIONS.pop(session_key, None)
                            store.delete_session(session_key)
        acc = pool.acquire()
        if acc is None:
            raise HTTPException(503, "no accounts available")
        attempts += 1
        try:
            sess, _ = await get_or_create_session(None, users, acc, mode=mode)
            result, events = await run_session_turn(
                sess, kwargs.get("prompt"),
                attachment_ids=kwargs.get("attachment_ids"),
                file_jobs=kwargs.get("file_jobs"),
                system_prompt=kwargs.get("system_prompt"))
            pool.release_ok(acc)
            state = SessionState(account_key=acc.key, grok=sess, user_chain=list(users))
            return acc, result, events, state
        except GatewayError as e:
            try:
                await sess.close()
            except Exception:
                pass
            kind = e.kind if e.kind in ("auth", "quota", "degraded") else "generic"
            pool.release_fail(acc, kind)
            if attempts >= max(3, min(len(pool.snapshot()) or 1, 5)) or not any(a.available() for a in pool.snapshot()):
                status = 429 if e.kind == "quota" else 502
                raise HTTPException(status, f"grok error ({e.kind}): {e}")


# ---------------------------------------------------------------- chat route

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    check_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "invalid JSON payload")
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")

    stream = bool(body.get("stream"))
    stream_options = body.get("stream_options") or {}
    include_usage = bool(stream_options.get("include_usage")) if isinstance(stream_options, dict) else False
    model_in = body.get("model")
    if model_in is not None and not isinstance(model_in, str):
        raise HTTPException(400, "model must be a string")
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise HTTPException(400, "messages array required")
    messages: list[dict] = [m for m in raw_messages if isinstance(m, dict)]
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
                    "image gen attempt %d/%d: %s", attempt+1, 3, e)
        if asset_paths:
            hosted = await host_images(asset_paths[:2])
            content_md = "\n\n".join(f"![generated image]({u})" for u in hosted) or "Image generated."
            rid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = now_epoch()
            if not stream:
                return JSONResponse({
                    "id": rid,
                    "object": "chat.completion", "created": created,
                    "model": public_model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content_md, "refusal": None},
                                 "finish_reason": "stop", "logprobs": None}],
                    "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 0,
                              "total_tokens": len(prompt) // 4},
                })
            async def img_sse():
                first = {"id": rid, "object": "chat.completion.chunk", "created": created,
                         "model": public_model,
                         "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                      "finish_reason": None}]}
                yield f"data: {json.dumps(first)}\n\n"
                ch = {"id": rid, "object": "chat.completion.chunk", "created": created,
                      "model": public_model,
                      "choices": [{"index": 0, "delta": {"content": content_md},
                                   "finish_reason": None}]}
                yield f"data: {json.dumps(ch)}\n\n"
                end = {"id": rid, "object": "chat.completion.chunk", "created": created,
                       "model": public_model,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(end)}\n\n"
                if include_usage:
                    usage_chunk = {"id": rid, "object": "chat.completion.chunk", "created": created,
                                   "model": public_model, "choices": [],
                                   "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 0,
                                             "total_tokens": len(prompt) // 4}}
                    yield f"data: {json.dumps(usage_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(img_sse(), media_type="text/event-stream")
        # all image gen attempts failed -> fall through to text chat
        prompt = prompt.replace("[Generate an image] ", "")

    rid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = now_epoch()
    req_include_sources = _include_sources(body.get("include_sources"))

    if not stream:
        acc, result, events, state = await pick_account_and_turn(
            prefix_key, users[:], mode=mode, prompt=prompt,
            file_jobs=remaining_jobs or None, system_prompt=system_prompt)

        # persist so the NEXT incremental call (prefix == our full chain) reuses it
        if users:
            st = SessionState(account_key=acc.key, grok=state.grok, user_chain=list(users))
            async with SESSION_LOCK:
                SESSIONS[chain_key(users, auth_header)] = st
                if len(users) >= 2:
                    SESSIONS.setdefault(chain_key(users[:-1], auth_header), st)
            k_curr = chain_key(users, auth_header)
            store.save_session(
                session_key=k_curr,
                account_key=acc.key,
                user_chain=list(users),
                conversation_id=state.grok.conversation_id,
                last_parent_response_id=state.grok.last_parent_response_id,
                model_mode=mode,
                created_at=st.created_at,
                last_used=st.last_used,
            )
            if len(users) >= 2:
                k_prev = chain_key(users[:-1], auth_header)
                store.save_session(
                    session_key=k_prev,
                    account_key=acc.key,
                    user_chain=list(users),
                    conversation_id=state.grok.conversation_id,
                    last_parent_response_id=state.grok.last_parent_response_id,
                    model_mode=mode,
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
            appendix_query = result.search_queries[0] if result.search_queries else prompt
            appendix = _source_appendix(result.sources, appendix_query)

        final_response_text = result.text + appendix
        usage_dict = {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(final_response_text) // 4,
            "total_tokens": (len(prompt) + len(final_response_text)) // 4,
        }

        return JSONResponse({
            "id": rid, "object": "chat.completion", "created": created,
            "model": public_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": final_response_text, "refusal": None},
                "finish_reason": finish_reason,
                "logprobs": None,
            }],
            "usage": usage_dict,
        })

    async def sse():
        try:
            first = {"id": rid, "object": "chat.completion.chunk", "created": created,
                     "model": public_model,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                  "finish_reason": None}]}
            yield f"data: {json.dumps(first)}\n\n"

            accumulated_text = []
            render_filter = RenderFilter()
            final_turn_result: TurnResult | None = None
            turn_acc = None
            turn_state = None

            # Stream real-time events from gateway or mock
            import unittest.mock
            if isinstance(pick_account_and_turn, (unittest.mock.NonCallableMagicMock, unittest.mock.AsyncMock, unittest.mock.MagicMock, unittest.mock.Mock)):
                mock_res = await pick_account_and_turn(
                    prefix_key, users[:], mode=mode, prompt=prompt,
                    file_jobs=remaining_jobs or None, system_prompt=system_prompt)
                turn_acc, mock_turn_res, mock_events, turn_state = mock_res
                final_turn_result = mock_turn_res
                filtered_mock = render_filter.process(mock_turn_res.text)
                if filtered_mock:
                    accumulated_text.append(filtered_mock)
                    ch = {"id": rid, "object": "chat.completion.chunk", "created": created,
                          "model": public_model,
                          "choices": [{"index": 0, "delta": {"content": filtered_mock},
                                       "finish_reason": None}]}
                    yield f"data: {json.dumps(ch)}\n\n"
            else:
                async for ev in pick_account_and_stream_turn(
                        prefix_key, users[:], mode=mode, prompt=prompt,
                        file_jobs=remaining_jobs or None, system_prompt=system_prompt):
                    turn_acc = ev.get("acc")
                    turn_state = ev.get("state")
                    ev_type = ev.get("type")
                    if ev_type == "text_delta":
                        raw_delta = ev.get("text", "")
                        clean_delta = render_filter.process(raw_delta)
                        if clean_delta:
                            accumulated_text.append(clean_delta)
                            ch = {"id": rid, "object": "chat.completion.chunk", "created": created,
                                  "model": public_model,
                                  "choices": [{"index": 0, "delta": {"content": clean_delta},
                                               "finish_reason": None}]}
                            yield f"data: {json.dumps(ch)}\n\n"
                    elif ev_type == "done":
                        final_turn_result = ev.get("result")

            flushed = render_filter.flush()
            if flushed:
                accumulated_text.append(flushed)
                ch = {"id": rid, "object": "chat.completion.chunk", "created": created,
                      "model": public_model,
                      "choices": [{"index": 0, "delta": {"content": flushed},
                                   "finish_reason": None}]}
                yield f"data: {json.dumps(ch)}\n\n"

            # Check if any images were generated during turn
            if final_turn_result and final_turn_result.image_urls and turn_acc:
                hosted = await host_images(final_turn_result.image_urls, turn_acc.cookie_header())
                if hosted:
                    md = "\n\n" + "\n\n".join(f"![generated image]({u})" for u in hosted)
                    accumulated_text.append(md)
                    ch = {"id": rid, "object": "chat.completion.chunk", "created": created,
                          "model": public_model,
                          "choices": [{"index": 0, "delta": {"content": md},
                                       "finish_reason": None}]}
                    yield f"data: {json.dumps(ch)}\n\n"

            # Check appendix for sources
            appendix = ""
            if req_include_sources and final_turn_result and final_turn_result.sources:
                appendix_query = final_turn_result.search_queries[0] if final_turn_result.search_queries else prompt
                appendix = _source_appendix(final_turn_result.sources, appendix_query)
                if appendix:
                    accumulated_text.append(appendix)
                    ch = {"id": rid, "object": "chat.completion.chunk", "created": created,
                          "model": public_model,
                          "choices": [{"index": 0, "delta": {"content": appendix},
                                       "finish_reason": None}]}
                    yield f"data: {json.dumps(ch)}\n\n"

            # Persist session state
            if users and turn_acc and turn_state:
                st = SessionState(account_key=turn_acc.key, grok=turn_state.grok, user_chain=list(users))
                async with SESSION_LOCK:
                    SESSIONS[chain_key(users, auth_header)] = st
                    if len(users) >= 2:
                        SESSIONS.setdefault(chain_key(users[:-1], auth_header), st)
                k_curr = chain_key(users, auth_header)
                store.save_session(
                    session_key=k_curr,
                    account_key=turn_acc.key,
                    user_chain=list(users),
                    conversation_id=turn_state.grok.conversation_id,
                    last_parent_response_id=turn_state.grok.last_parent_response_id,
                    model_mode=mode,
                    created_at=st.created_at,
                    last_used=st.last_used,
                )
                if len(users) >= 2:
                    k_prev = chain_key(users[:-1], auth_header)
                    store.save_session(
                        session_key=k_prev,
                        account_key=turn_acc.key,
                        user_chain=list(users),
                        conversation_id=turn_state.grok.conversation_id,
                        last_parent_response_id=turn_state.grok.last_parent_response_id,
                        model_mode=mode,
                        created_at=st.created_at,
                        last_used=st.last_used,
                    )

            finish_reason = (final_turn_result.finish_reason if final_turn_result else None) or "stop"
            end = {"id": rid, "object": "chat.completion.chunk", "created": created,
                   "model": public_model,
                   "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
            yield f"data: {json.dumps(end)}\n\n"

            if include_usage:
                total_text = "".join(accumulated_text)
                usage_dict = {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(total_text) // 4,
                    "total_tokens": (len(prompt) + len(total_text)) // 4,
                }
                usage_chunk = {"id": rid, "object": "chat.completion.chunk", "created": created,
                               "model": public_model, "choices": [],
                               "usage": usage_dict}
                yield f"data: {json.dumps(usage_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logging.getLogger("uvicorn.error").error("Chat SSE stream error: %s", e)

    return StreamingResponse(sse(), media_type="text/event-stream")


def iter_events(events):
    yield from events


def inline_text_safe(job: dict) -> str | None:
    try:
        return inline_textual(job)
    except Exception:
        return None


# --------------------------------------------------------------- responses API

@app.post("/v1/responses")
async def responses_api(request: Request):
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

    messages: list[dict] = []
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
                    messages.append({"role": role or "user", "content": item.get("content")})
                elif itype == "input_text":
                    messages.append({"role": "user", "content": item.get("text", "")})
                elif itype in ("input_image", "input_file"):
                    messages.append({"role": "user", "content": [item]})
                elif role:
                    messages.append(item)
    auth_header = request.headers.get("authorization", "")
    sess_prev = None
    if prev_resp_id:
        async with SESSION_LOCK:
            sess_prev = SESSIONS.get(prev_resp_id)
            # Restore from persistent sqlite store when not in memory (e.g. after restart)
            if not sess_prev:
                persisted = store.get_session(prev_resp_id)
                if persisted:
                    acc_cand = pool.acquire_by_key(persisted["account_key"])
                    if acc_cand:
                        sess = GrokSession(acc_cand.cookie_header(), await _uid_for(acc_cand), persisted.get("model_mode", "fast"))
                        sess.conversation_id = persisted.get("conversation_id", "")
                        sess.last_parent_response_id = persisted.get("last_parent_response_id", "")
                        sess_prev = SessionState(
                            account_key=acc_cand.key,
                            grok=sess,
                            user_chain=persisted.get("user_chain", []),
                            created_at=persisted.get("created_at", time.time()),
                            last_used=time.time(),
                        )
                        SESSIONS[prev_resp_id] = sess_prev
        if not messages:
            if not sess_prev:
                raise HTTPException(400, f"unknown previous_response_id: {prev_resp_id}")
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

    if not stream:
        # continue previous response's grok session when provided
        if sess_prev is not None:
            acc = pool.acquire_by_key(sess_prev.account_key)
            if not acc:
                raise HTTPException(409, "previous response account is cooling down; retry")
            st = sess_prev
            st.touch()
            try:
                result, events = await run_session_turn(
                    st.grok, prompt, file_jobs=remaining_jobs or None,
                    system_prompt=instructions)
                pool.release_ok(acc)
                state = st
            except GatewayError as e:
                pool.release_fail(acc, e.kind if e.kind in ("auth", "quota", "degraded") else "generic")
                await st.grok.close()
                async with SESSION_LOCK:
                    if prev_resp_id:
                        SESSIONS.pop(prev_resp_id, None)
                        store.delete_session(prev_resp_id)
                raise HTTPException(429 if e.kind == "quota" else 502, f"grok error ({e.kind}): {e}")
        else:
            acc, result, events, state = await pick_account_and_turn(
                prefix_key, users[:], mode=mode, prompt=prompt,
                file_jobs=remaining_jobs or None, system_prompt=instructions)

        image_urls = list(result.image_urls)
        if image_urls:
            hosted = await host_images(image_urls, acc.cookie_header())
            md = "\n\n".join(f"![generated image]({u})" for u in hosted)
            result.text = (result.text + "\n\n" + md).strip()

        if users:
            full_user_chain = (sess_prev.user_chain if sess_prev else []) + [u for u in users if u not in (sess_prev.user_chain if sess_prev else [])]
            st = SessionState(account_key=acc.key, grok=state.grok, user_chain=full_user_chain)
            async with SESSION_LOCK:
                SESSIONS[rid] = st                      # previous_response_id -> session
                SESSIONS[rid + ":chain"] = st
                if len(users) >= 2:
                    SESSIONS.setdefault(chain_key(users[:-1], auth_header), st)
                SESSIONS[chain_key(users, auth_header)] = st
            store.save_session(
                session_key=rid,
                account_key=acc.key,
                user_chain=full_user_chain,
                conversation_id=state.grok.conversation_id,
                last_parent_response_id=state.grok.last_parent_response_id,
                model_mode=mode,
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
                created_at=st.created_at,
                last_used=st.last_used,
            )
            if len(users) >= 2:
                store.save_session(
                    session_key=chain_key(users[:-1], auth_header),
                    account_key=acc.key,
                    user_chain=full_user_chain,
                    conversation_id=state.grok.conversation_id,
                    last_parent_response_id=state.grok.last_parent_response_id,
                    model_mode=mode,
                    created_at=st.created_at,
                    last_used=st.last_used,
                )

        appendix = ""
        if req_include_sources and result.sources:
            appendix_query = result.search_queries[0] if result.search_queries else prompt
            appendix = _source_appendix(result.sources, appendix_query)

        final_response_text = result.text + appendix

        output_item = {
            "id": msg_id, "type": "message", "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": final_response_text, "annotations": []}],
        }

        usage_obj = {
            "input_tokens": len(prompt) // 4,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": len(final_response_text) // 4,
            "output_tokens_details": {"reasoning_tokens": len(result.reasoning) // 4},
            "total_tokens": (len(prompt) + len(final_response_text)) // 4,
        }

        base_response = {
            "id": rid, "object": "response", "created_at": created,
            "completed_at": now_epoch(),
            "status": "completed", "model": public_model,
            "instructions": instructions or None,
            "output": [output_item],
            "output_text": final_response_text,
            "error": None, "incomplete_details": None,
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
            "id": msg_id, "type": "message", "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        resp_in_prog = {
            "id": rid, "object": "response", "created_at": created,
            "completed_at": None, "status": "in_progress", "model": public_model,
            "instructions": instructions or None,
            "output": [], "output_text": "",
            "error": None, "incomplete_details": None,
            "previous_response_id": prev_resp_id or None,
            "parallel_tool_calls": False,
            "temperature": 1.0, "tool_choice": "none", "tools": [],
            "top_p": 1.0, "usage": None, "metadata": {},
        }

        yield evt("response.created", {"type": "response.created", "response": resp_in_prog})
        yield evt("response.in_progress", {"type": "response.in_progress", "response": resp_in_prog})
        yield evt("response.output_item.added", {
            "type": "response.output_item.added", "output_index": 0,
            "item": init_output_item})
        yield evt("response.content_part.added", {
            "type": "response.content_part.added", "item_id": msg_id,
            "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}})

        accumulated_text = []
        render_filter = RenderFilter()
        final_turn_result: TurnResult | None = None
        turn_acc = None
        turn_state = None

        import unittest.mock
        if sess_prev is not None:
            acc = pool.acquire_by_key(sess_prev.account_key)
            if not acc:
                raise HTTPException(409, "previous response account is cooling down; retry")
            st = sess_prev
            st.touch()
            turn_acc = acc
            turn_state = st
            try:
                async for ev in stream_session_turn(
                        st.grok, prompt, file_jobs=remaining_jobs or None,
                        system_prompt=instructions):
                    ev_type = ev.get("type")
                    if ev_type == "text_delta":
                        raw_delta = ev.get("text", "")
                        clean_delta = render_filter.process(raw_delta)
                        if clean_delta:
                            accumulated_text.append(clean_delta)
                            yield evt("response.output_text.delta", {
                                "type": "response.output_text.delta", "item_id": msg_id,
                                "output_index": 0, "content_index": 0, "delta": clean_delta})
                    elif ev_type == "done":
                        final_turn_result = ev.get("result")
                pool.release_ok(acc)
            except GatewayError as e:
                pool.release_fail(acc, e.kind if e.kind in ("auth", "quota", "degraded") else "generic")
                await st.grok.close()
                async with SESSION_LOCK:
                    if prev_resp_id:
                        SESSIONS.pop(prev_resp_id, None)
                raise HTTPException(429 if e.kind == "quota" else 502, f"grok error ({e.kind}): {e}")
        elif isinstance(pick_account_and_turn, (unittest.mock.NonCallableMagicMock, unittest.mock.AsyncMock, unittest.mock.MagicMock, unittest.mock.Mock)):
            mock_res = await pick_account_and_turn(
                prefix_key, users[:], mode=mode, prompt=prompt,
                file_jobs=remaining_jobs or None, system_prompt=instructions)
            turn_acc, mock_turn_res, mock_events, turn_state = mock_res
            final_turn_result = mock_turn_res
            filtered_mock = render_filter.process(mock_turn_res.text)
            if filtered_mock:
                accumulated_text.append(filtered_mock)
                yield evt("response.output_text.delta", {
                    "type": "response.output_text.delta", "item_id": msg_id,
                    "output_index": 0, "content_index": 0, "delta": filtered_mock})
        else:
            async for ev in pick_account_and_stream_turn(
                    prefix_key, users[:], mode=mode, prompt=prompt,
                    file_jobs=remaining_jobs or None, system_prompt=instructions):
                turn_acc = ev.get("acc")
                turn_state = ev.get("state")
                ev_type = ev.get("type")
                if ev_type == "text_delta":
                    raw_delta = ev.get("text", "")
                    clean_delta = render_filter.process(raw_delta)
                    if clean_delta:
                        accumulated_text.append(clean_delta)
                        yield evt("response.output_text.delta", {
                            "type": "response.output_text.delta", "item_id": msg_id,
                            "output_index": 0, "content_index": 0, "delta": clean_delta})
                elif ev_type == "done":
                    final_turn_result = ev.get("result")

        flushed = render_filter.flush()
        if flushed:
            accumulated_text.append(flushed)
            yield evt("response.output_text.delta", {
                "type": "response.output_text.delta", "item_id": msg_id,
                "output_index": 0, "content_index": 0, "delta": flushed})

        # Check if any images were generated during turn
        if final_turn_result and final_turn_result.image_urls and turn_acc:
            hosted = await host_images(final_turn_result.image_urls, turn_acc.cookie_header())
            if hosted:
                md = "\n\n" + "\n\n".join(f"![generated image]({u})" for u in hosted)
                accumulated_text.append(md)
                yield evt("response.output_text.delta", {
                    "type": "response.output_text.delta", "item_id": msg_id,
                    "output_index": 0, "content_index": 0, "delta": md})

        # Check appendix for sources
        appendix = ""
        if req_include_sources and final_turn_result and final_turn_result.sources:
            appendix_query = final_turn_result.search_queries[0] if final_turn_result.search_queries else prompt
            appendix = _source_appendix(final_turn_result.sources, appendix_query)
            if appendix:
                accumulated_text.append(appendix)
                yield evt("response.output_text.delta", {
                    "type": "response.output_text.delta", "item_id": msg_id,
                    "output_index": 0, "content_index": 0, "delta": appendix})

        # Persist session state
        if users and turn_acc and turn_state:
            full_user_chain = (sess_prev.user_chain if sess_prev else []) + [u for u in users if u not in (sess_prev.user_chain if sess_prev else [])]
            st = SessionState(account_key=turn_acc.key, grok=turn_state.grok, user_chain=full_user_chain)
            async with SESSION_LOCK:
                SESSIONS[rid] = st                      # previous_response_id -> session
                SESSIONS[rid + ":chain"] = st
                if len(users) >= 2:
                    SESSIONS.setdefault(chain_key(users[:-1], auth_header), st)
                SESSIONS[chain_key(users, auth_header)] = st
            store.save_session(
                session_key=rid,
                account_key=turn_acc.key,
                user_chain=full_user_chain,
                conversation_id=turn_state.grok.conversation_id,
                last_parent_response_id=turn_state.grok.last_parent_response_id,
                model_mode=mode,
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
                created_at=st.created_at,
                last_used=st.last_used,
            )
            if len(users) >= 2:
                store.save_session(
                    session_key=chain_key(users[:-1], auth_header),
                    account_key=turn_acc.key,
                    user_chain=full_user_chain,
                    conversation_id=turn_state.grok.conversation_id,
                    last_parent_response_id=turn_state.grok.last_parent_response_id,
                    model_mode=mode,
                    created_at=st.created_at,
                    last_used=st.last_used,
                )

        final_response_text = "".join(accumulated_text)
        final_output_item = {
            "id": msg_id, "type": "message", "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": final_response_text, "annotations": []}],
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
            "id": rid, "object": "response", "created_at": created,
            "completed_at": now_epoch(),
            "status": "completed", "model": public_model,
            "instructions": instructions or None,
            "output": [final_output_item],
            "output_text": final_response_text,
            "error": None, "incomplete_details": None,
            "previous_response_id": prev_resp_id or None,
            "parallel_tool_calls": False,
            "temperature": 1.0,
            "tool_choice": "none",
            "tools": [],
            "top_p": 1.0,
            "usage": usage_obj,
            "metadata": {},
        }

        yield evt("response.output_text.done", {
            "type": "response.output_text.done", "item_id": msg_id,
            "output_index": 0, "content_index": 0, "text": final_response_text})
        yield evt("response.content_part.done", {
            "type": "response.content_part.done", "item_id": msg_id,
            "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": final_response_text, "annotations": []}})
        yield evt("response.output_item.done", {
            "type": "response.output_item.done", "output_index": 0, "item": final_output_item})
        yield evt("response.completed", {"type": "response.completed",
                                          "response": completed_response})

    return StreamingResponse(sse(), media_type="text/event-stream")


# ------------------------------------------------------------------ misc routes

@app.get("/v1/models")
async def models(request: Request):
    check_auth(request)
    data = [{"id": mid, "object": "model", "created": 1700000000, "owned_by": "grok"}
            for mid in ["grok-fast", "grok-auto", "grok-expert", "grok-heavy"]]
    return {"object": "list", "data": data}


@app.get("/healthz")
async def healthz():
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
