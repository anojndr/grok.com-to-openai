"""Grok WebSocket Gateway client (wss://grok.com/ws/mgw/).

A GrokSession wraps one live gateway connection = one grok conversation.
Sequential ask() calls on the same session are proper multi-turn: grok keeps
the full context server-side, only the new user message travels the wire.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import websockets

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

from config import GROK_BASE, USER_AGENT

ASSET_BASE = "https://assets.grok.com/"

RENDER_TAG_RE = re.compile(r"<grok:render\b.*?(?:</grok:render>|$)", re.DOTALL)
URL_OR_PATH_RE = re.compile(
    r'(?:https?://assets\.grok\.com/users/[^\s"\'<>]+\.(?:jpg|jpeg|png|webp|gif)|users/[0-9a-fA-F-]+/(?:generated|attachments)/[^\s"\'<>]+\.(?:jpg|jpeg|png|webp|gif))',
    re.IGNORECASE,
)


def strip_render_tags(text: str) -> str:
    """Remove grok:render pseudo-HTML card markup from streamed text."""
    return RENDER_TAG_RE.sub("", text)


class RenderFilter:
    """Incremental filter to strip <grok:render>...</grok:render> tags from streaming text deltas."""

    def __init__(self) -> None:
        self._buffer: str = ""
        self._in_tag: bool = False

    def process(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buffer += chunk
        out: list[str] = []
        while self._buffer:
            if not self._in_tag:
                idx = self._buffer.find("<grok:render")
                if idx == -1:
                    # Check for potential partial prefix of "<grok:render" at the tail
                    tag_start = "<grok:render"
                    partial_len = 0
                    for k in range(1, min(len(tag_start), len(self._buffer) + 1)):
                        if tag_start.startswith(self._buffer[-k:]):
                            partial_len = k
                    if partial_len > 0:
                        safe = self._buffer[:-partial_len]
                        self._buffer = self._buffer[-partial_len:]
                        if safe:
                            out.append(safe)
                        break
                    else:
                        out.append(self._buffer)
                        self._buffer = ""
                        break
                else:
                    if idx > 0:
                        out.append(self._buffer[:idx])
                        self._buffer = self._buffer[idx:]
                    self._in_tag = True
            else:
                # In tag: look for closing tag </grok:render>
                close_idx = self._buffer.find("</grok:render>")
                if close_idx == -1:
                    # We are still inside the tag, buffer everything inside
                    break
                else:
                    # Skip past </grok:render>
                    self._buffer = self._buffer[close_idx + len("</grok:render>") :]
                    self._in_tag = False
        return "".join(out)

    def flush(self) -> str:
        # At the end of turn, if we were not inside an actual render tag, flush remaining buffer
        if not self._in_tag:
            res = self._buffer
            self._buffer = ""
            return res
        self._buffer = ""
        return ""


def _asset_url(u: str) -> str:
    if u.startswith(("data:", "http://", "https://")):
        return u
    return ASSET_BASE + u.lstrip("/")


def chunk_image_url(value: Any) -> str | None:
    """Extract a finished image URL/data-URI from a gateway chunk payload.

    Image generation/editing arrives as card chunks — render_edited_image /
    render_generated_image with an inner image_chunk ({imageUrl, progress};
    progress 50 carries a data: URI placeholder, 100 the final asset path),
    plus legacy flat shapes ({imageUrl|url|image_url}). Returns None for
    in-progress placeholders and non-image payloads.
    """
    if not isinstance(value, dict):
        return None
    ic = value.get("image_chunk")
    if isinstance(ic, dict):
        u = ic.get("imageUrl") or ic.get("image_url") or ""
        prog = ic.get("progress")
        if isinstance(u, str) and u and (prog is None or prog >= 100):
            return _asset_url(u)
        return None
    u = value.get("imageUrl") or value.get("url") or value.get("image_url")
    if isinstance(u, str) and u:
        return _asset_url(u)
    # Card payloads sometimes carry the asset path in an unexpected field;
    # scan the values for a Grok asset path before giving up.
    for v in value.values():
        if isinstance(v, str) and (
            "assets.grok.com/users/" in v or v.startswith("users/")
        ):
            m = URL_OR_PATH_RE.search(v)
            if m:
                return _asset_url(m.group(0))
    return None


def _card_query(card: Any) -> str | None:
    """Extract the search query from a tool_usage_card payload.

    Live mgw frames carry it on response.chunk as
    tool_usage_card.web_search.args.query; tolerate the flat .query variant
    and the response.grok.output placement seen on other gateway builds.
    """
    if not isinstance(card, dict):
        return None
    wq = card.get("web_search")
    if not isinstance(wq, dict):
        return None
    args = wq.get("args")
    q = args.get("query") if isinstance(args, dict) else None
    if not isinstance(q, str):
        q = wq.get("query")
    if not isinstance(q, str) or not q:
        return None
    stripped = q.strip()
    if not isinstance(stripped, str) or not stripped:
        return None
    return stripped


def extract_web_results(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize citation entries ({url, title}) out of a gateway event.

    Multiple event shapes carry grok's web citations:
      response.chunk        -> chunk.tool_result.web_search.webpages[]
      response.chunk        -> chunk.tool_result.web_search_results[] / web_results[]
      response.grok.output  -> output.tool_result.web_search.webpages[]
      response.grok.output  -> output.card_attachment {url, title}
      response.search.result -> result.web_results[] / result.webSearch_results[]
    """
    if not isinstance(event, dict):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_entry(item: Any) -> None:
        if not isinstance(item, dict):
            return
        url = str(item.get("url") or "").strip()
        if not url:
            return
        key = url.lower()
        if key in seen:
            return
        seen.add(key)
        title = str(item.get("title") or "").strip() or url
        out.append({"url": url, "title": title})

    def scan_obj(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        # check direct card_attachment
        card_att = obj.get("card_attachment")
        if isinstance(card_att, dict):
            add_entry(card_att)

        # check result payload
        result = obj.get("result")
        if isinstance(result, dict):
            for k in ("web_results", "webSearch_results", "webpages", "results"):
                items = result.get(k)
                if isinstance(items, list):
                    for item in items:
                        add_entry(item)

        # check tool_result payload
        tool = obj.get("tool_result")
        if isinstance(tool, dict):
            ws = tool.get("web_search")
            if isinstance(ws, dict):
                for k in ("webpages", "results", "web_results"):
                    items = ws.get(k)
                    if isinstance(items, list):
                        for item in items:
                            add_entry(item)
            for k in ("web_search_results", "web_results", "webpages", "results"):
                items = tool.get(k)
                if isinstance(items, list):
                    for item in items:
                        add_entry(item)

    sub = event.get("event") if isinstance(event.get("event"), dict) else {}
    for container in (event, sub):
        if not isinstance(container, dict):
            continue
        scan_obj(container)
        scan_obj(container.get("output"))
        scan_obj(container.get("chunk"))

    return out


class GatewayError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# Degraded-turn detection
# ---------------------------------------------------------------------------
# Some accounts' gateways intermittently run their web-search tool against
# placeholder queries unrelated to the user's message (observed live:
# "current information and recent sources", "best expert recommendations
# and evidence", "latest updates and authoritative references"), then the
# model summarizes that unrelated content into fluent-looking word salad
# that still parses as a successful 200. The turn is detectable before the
# salad streams: tool_usage_card.web_search.query arrives on
# response.grok.output ahead of the text deltas. A genuine search almost
# always echoes at least one significant token of the user's message, so a
# query sharing zero tokens is the signature of the degraded path.

USER_TOKEN_RE = re.compile(r"[a-z0-9]{4,}")
# Observed placeholder queries from degraded gateways; a single match here
# is enough to fail the turn even when only one search has run.
GENERIC_SEARCH_QUERIES = frozenset(
    {
        "current information and recent sources",
        "best expert recommendations and evidence",
        "latest updates and authoritative references",
    }
)


STOPWORD_TOKENS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "any",
        "can",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "man",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "its",
        "did",
        "that",
        "she",
        "they",
        "with",
        "what",
        "when",
        "where",
        "which",
        "this",
        "from",
        "have",
        "will",
        "your",
        "into",
    }
)


def unrelated_queries(queries: list[str], user_text: str) -> bool:
    """True when 2+ tool queries share no token with the user text, or any query is a known placeholder.

    A known placeholder query aborts immediately on its own, even if the prompt is short or empty.
    When user_text has tokens, 2 or more unrelated queries (whether distinct or duplicated)
    signal degraded behavior. Duplicated identical queries count once toward that threshold;
    a single unrelated query is tolerated as legitimate paraphrase.
    """
    if not queries:
        return False
    cleaned = [q.lower().strip() for q in queries if q and q.lower().strip()]
    if any(q in GENERIC_SEARCH_QUERIES for q in cleaned):
        return True
    if not user_text:
        return False
    raw_tokens = re.findall(r"[a-z0-9]{3,}|[^\x00-\x7f]+", user_text.lower())
    tokens = {t for t in raw_tokens if t not in STOPWORD_TOKENS}
    if not tokens:
        return False
    unrelated = [q for q in cleaned if not any(t in q for t in tokens)]
    # 2+ unrelated queries signal degraded behavior; repeated identical unrelated queries
    # also count (the gateway is stuck searching the same off-topic content).
    distinct_unrelated = {q for q in unrelated}
    if len(distinct_unrelated) >= 2:
        return True
    return len(unrelated) >= 2


@dataclass
class TurnResult:
    text: str = ""
    reasoning: str = ""
    image_urls: list[str] = field(default_factory=list)
    # parallel to image_urls: "edited" | "generated" | "unknown".
    # First-seen-wins: the first card key that surfaces a URL labels it, so a
    # URL first seen under a generic/generated key is never re-labeled
    # "edited" by a later edit card. Healthy gateways stream edits under
    # render_edited_image (live captures 2026-08-31) — the assumption the
    # degraded-edit guard in ask() rests on.
    image_kinds: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    response_id: str = ""
    conversation_id: str = ""
    parent_response_id: str = ""
    finish_reason: str = "stop"


def new_uuid() -> str:
    return str(uuid.uuid4())


def default_x_grok() -> dict[str, Any]:
    return {
        "protocol_capabilities": ["conversation_attached", "custom_methods_v1"],
        "use_chunk": True,
        "enable_side_by_side": True,
        "force_side_by_side": False,
        "enable_image_generation": True,
        "image_generation_count": 2,
        "disable_text_follow_ups": False,
        "disable_artifact": True,
        "force_concise": False,
        # keep_context MUST stay True: continued turns send only the newest
        # user message and rely on the gateway holding the conversation.
        # (False caused total multi-turn amnesia: "remember 974" followed by
        # "what number?" hallucinated "3".) is_temporary/disable_memory stay
        # set so pooled accounts never persist chats or leak long-term memory
        # across unrelated users sharing an account.
        "keep_context": True,
        "is_temporary": True,
        "disable_memory": True,
    }


async def resolve_user_id(cookie_header: str) -> str:
    from curl_cffi.requests import AsyncSession

    async with AsyncSession[Any](impersonate="chrome") as s:
        r = await s.get(
            f"{GROK_BASE}/api/auth/session",
            headers={"user-agent": USER_AGENT, "cookie": cookie_header},
            timeout=10,
        )
        if r.status_code == 200:
            try:
                data = r.json()
            except (ValueError, TypeError, AttributeError):
                data = None
            if isinstance(data, dict):
                user = data.get("user")
                user_id = user.get("id") if isinstance(user, dict) else None
                session = data.get("session")
                session_user_id = (
                    session.get("userId") if isinstance(session, dict) else None
                )
                uid = user_id or session_user_id or data.get("userId") or ""
                if isinstance(uid, str) and uid:
                    return uid
        raise GatewayError("auth", f"failed to resolve user id ({r.status_code})")


class GrokSession:
    def __init__(self, cookie_header: str, user_id: str, model_mode: str = "fast"):
        self.cookie_header, self.user_id, self.model_mode = (
            cookie_header,
            user_id,
            model_mode,
        )
        self.ws: ClientConnection | None = None
        self.ws_mode = model_mode  # mode the live ws session was negotiated with
        self.conversation_id = ""
        self.last_parent_response_id = ""
        # Uploaded attachments made visible on this conversation, oldest first:
        # [{"file_id": <grok fileMetadataId>, "hash": <sha256 hex | None>}, ...].
        # The gateway only "sees" files mentioned on the CURRENT message, so
        # later turns re-mention these ids (see server.stream_session_turn).
        self.attachments: list[dict[str, Any]] = []
        # Ids stream_session_turn declared stale on the most recent turn
        # (reset every turn); pick_account_and_* propagate them to the source
        # checkpoint so failover does not re-mention dead files.
        self.last_dropped_attachment_ids: set[str] = set()
        self.lock = asyncio.Lock()

    def clone_checkpoint(self, model_mode: str | None = None) -> GrokSession:
        """Create a disconnected session positioned at this conversation checkpoint."""
        sess = GrokSession(
            self.cookie_header,
            self.user_id,
            model_mode or self.model_mode,
        )
        sess.conversation_id = self.conversation_id
        sess.last_parent_response_id = self.last_parent_response_id
        sess.attachments = [dict(a) for a in self.attachments]
        return sess

    async def connect(self) -> None:
        # A reconnect (e.g. model-mode switch on a warm socket) must not
        # orphan the previous live connection.
        await self.close()
        uri = f"wss://grok.com/ws/mgw/?uid={self.user_id}"
        headers = {
            "Origin": GROK_BASE,
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": self.cookie_header,
        }
        try:
            self.ws = await websockets.connect(
                uri,
                additional_headers=headers,
                max_size=32 * 1024 * 1024,
                open_timeout=10,
                close_timeout=5,
            )
            ws = self.ws
            if ws is None:
                raise GatewayError(
                    "upstream", "connect failed: no connection established"
                )
            xgrok = default_x_grok()
            if self.conversation_id:
                xgrok["conversation_id"] = self.conversation_id
            await ws.send(
                json.dumps(
                    {
                        "event": {
                            "type": "session.create",
                            "event_id": "evt_init_" + new_uuid(),
                            "session": {"model": self.model_mode, "x_grok": xgrok},
                        }
                    }
                )
            )
            got_session_id = False
            while not got_session_id:
                raw = await asyncio.wait_for(ws.recv(), timeout=15)
                try:
                    env = json.loads(raw)
                except (ValueError, TypeError, AttributeError) as e:
                    raise GatewayError(
                        "upstream", f"malformed json in handshake: {e}"
                    ) from e
                ev = env.get("event") or {}
                if ev.get("type") == "session.created":
                    if not self.conversation_id:
                        self.conversation_id = env.get("session_id") or ""
                    self.ws_mode = self.model_mode
                    got_session_id = True
                elif ev.get("type") == "error":
                    raise GatewayError("upstream", json.dumps(ev.get("error"))[:200])
        except GatewayError:
            await self.close()
            raise
        except TimeoutError as e:
            await self.close()
            raise GatewayError(
                "timeout", "gateway connect or handshake timed out"
            ) from e
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as e:
            await self.close()
            # Raw socket/TLS errors must fail over like any other retryable
            # gateway failure instead of escaping as a 500.
            raise GatewayError(
                "upstream", f"connect failed: {type(e).__name__}: {e}"
            ) from e

    async def close(self) -> None:
        # Prune and fork race on the live WebSocket: fork moves ws under
        # source.lock, while prune pops from SESSIONS under SESSION_LOCK.
        # Close must not race the hand-off, but ask() already holds self.lock
        # when it calls close() on failure, so avoid deadlock by not
        # re-acquiring when already locked.
        if self.lock.locked():
            ws = self.ws
            self.ws = None
            if ws:
                with suppress(OSError, RuntimeError, AttributeError):
                    await ws.close()
            return
        async with self.lock:
            if self.ws:
                with suppress(OSError, RuntimeError, AttributeError):
                    await self.ws.close()
                self.ws = None

    def alive(self) -> bool:
        ws = self.ws
        if ws is None:
            return False
        try:
            from websockets.protocol import State

            state = getattr(getattr(ws, "protocol", ws), "state", None)
            if isinstance(state, State):
                return state is State.OPEN
            return False
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False

    async def ask(
        self,
        prompt: str,
        *,
        attachment_ids: list[str] | None = None,
        system_prompt: str | None = None,
        user_text: str = "",
        idle_timeout: float = 120.0,
        max_turn_timeout: float = 300.0,
    ) -> AsyncIterator[dict[str, Any]]:
        async with self.lock:
            completed_turn = False
            try:
                if not self.alive() or self.ws_mode != self.model_mode:
                    await self.connect()
                ws = self.ws
                if ws is None:
                    raise GatewayError("closed", "connection unavailable for turn")
                chunks = []
                if system_prompt:
                    chunks.append({"text": {"text": system_prompt + "\n\n"}})
                if attachment_ids:
                    for aid in attachment_ids:
                        chunks.append({"mention": {"file_mention": {"file_id": aid}}})
                chunks.append({"text": {"text": prompt}})
                now_ms = int(time.time() * 1000)
                item = {
                    "type": "message",
                    "role": "user",
                    "x_grok": {"client_message_id": new_uuid(), "input_chunks": chunks},
                }
                item_ev = {
                    "session_id": self.conversation_id,
                    "event": {
                        "type": "conversation.item.create",
                        "event_id": f"evt_msg_{now_ms}_{new_uuid()[:8]}",
                        "item": item,
                    },
                }
                if self.last_parent_response_id:
                    item_ev["event"]["parent_response_id"] = (
                        self.last_parent_response_id
                    )
                await ws.send(json.dumps(item_ev))
                await ws.send(
                    json.dumps(
                        {
                            "session_id": self.conversation_id,
                            "event": {
                                "type": "response.create",
                                "event_id": f"evt_resp_{now_ms}_{new_uuid()[:8]}",
                            },
                        }
                    )
                )

                text, reasoning, images, kinds = [], [], [], []
                tool_queries: list[str] = []  # web_search queries seen this turn
                sources: list[dict[str, Any]] = []
                seen_source_urls: set[str] = set()
                response_id, user_msg_id = "", ""
                turn_start = time.time()
                last = turn_start
                while True:
                    if time.time() - turn_start > max_turn_timeout:
                        raise GatewayError(
                            "timeout", "gateway turn exceeded max duration"
                        )
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=max(1.0, idle_timeout - (time.time() - last)),
                        )
                        last = time.time()
                    except TimeoutError:
                        raise GatewayError("timeout", "gateway idle timeout")
                    except websockets.ConnectionClosed:
                        raise GatewayError("closed", "connection closed mid-turn")
                    try:
                        env = json.loads(raw)
                    except (ValueError, TypeError, AttributeError) as e:
                        logging.getLogger("uvicorn.error").debug(
                            "skipping malformed gateway frame: %s", e
                        )
                        continue
                    ev = env.get("event") or {}
                    et = ev.get("type", "")
                    if et == "error":
                        err = ev.get("error") or {}
                        raise GatewayError(
                            "upstream", err.get("message") or json.dumps(err)[:200]
                        )
                    elif et == "response.search.result":
                        for src in extract_web_results(env):
                            k = src["url"].lower()
                            if k not in seen_source_urls:
                                seen_source_urls.add(k)
                                sources.append(src)
                    elif et == "response.grok.output":
                        out = ev.get("output") or {}
                        serr = out.get("stream_error")
                        if serr:
                            kind = serr.get("kind", "")
                            raise GatewayError(
                                "quota" if "usage_limit" in kind else "upstream",
                                serr.get("message", "stream error"),
                            )
                        query = _card_query(out.get("tool_usage_card"))
                        if query:
                            tool_queries.append(query)
                            if unrelated_queries(tool_queries, user_text):
                                raise GatewayError(
                                    "degraded",
                                    "gateway searched content unrelated to the "
                                    "request (degraded account)",
                                )
                        for src in extract_web_results(env):
                            k = src["url"].lower()
                            if k not in seen_source_urls:
                                seen_source_urls.add(k)
                                sources.append(src)
                        for obj in (out.get("generated_image"), out.get("image")):
                            u = chunk_image_url(obj)
                            if u and u not in images:
                                images.append(u)
                                kinds.append("generated")
                    elif et == "response.chunk":
                        chunk = ev.get("chunk") or {}
                        query = _card_query(chunk.get("tool_usage_card"))
                        if query:
                            tool_queries.append(query)
                            if unrelated_queries(tool_queries, user_text):
                                raise GatewayError(
                                    "degraded",
                                    "gateway searched content unrelated to the "
                                    "request (degraded account)",
                                )
                        for src in extract_web_results(env):
                            k = src["url"].lower()
                            if k not in seen_source_urls:
                                seen_source_urls.add(k)
                                sources.append(src)
                        tinfo = chunk.get("text") or {}
                        val = tinfo.get("text", "")
                        channel = tinfo.get("channel", "")
                        for key in (
                            "render_edited_image",
                            "render_generated_image",
                            "imageChunk",
                            "image_attachment",
                            "imageAttachment",
                            "image",
                            "generatedImage",
                            "media",
                        ):
                            u = chunk_image_url(chunk.get(key))
                            if u and u not in images:
                                images.append(u)
                                kinds.append(
                                    "edited"
                                    if key == "render_edited_image"
                                    else "generated"
                                )
                        if val:
                            if "NOTETAKER_HEADER" in channel:
                                # Timeline title chrome ("Thinking about your
                                # request", "Writing a ... story"): neither answer
                                # text nor reasoning. Live healthy fast turns
                                # stream it, so it must not pollute reasoning
                                # nor trip any degraded detector.
                                continue
                            if "THINKING" in channel or "NOTETAKER" in channel:
                                # Thinking/summary side-channels; bare NOTETAKER
                                # is the legacy wire name kept for old captures
                                # (see test_reasoning_with_generated_image_completes).
                                reasoning.append(val)
                                yield {"type": "reasoning_delta", "text": val}
                            else:
                                text.append(val)
                                yield {"type": "text_delta", "text": val}
                    elif et == "response.output_text.delta":
                        d = ev.get("delta")
                        if d:
                            text.append(d)
                            yield {"type": "text_delta", "text": d}
                    elif et == "response.reasoning_text.delta":
                        d = ev.get("delta", "")
                        if d:
                            reasoning.append(d)
                            yield {"type": "reasoning_delta", "text": d}
                    elif et == "conversation.item.added":
                        it = ev.get("item") or {}
                        if it.get("role") == "user":
                            user_msg_id = it.get("id") or user_msg_id
                    elif et == "response.output_item.added":
                        it = ev.get("item") or {}
                        if it.get("role") == "assistant":
                            response_id = it.get("id") or response_id
                    elif et == "response.done":
                        resp = ev.get("response") or {}
                        status = resp.get("status", "completed")
                        raw_text = "".join(text)
                        for match in URL_OR_PATH_RE.findall(raw_text):
                            img_u = _asset_url(match)
                            if img_u not in images:
                                images.append(img_u)
                                kinds.append("unknown")
                        result = TurnResult(
                            text=strip_render_tags(raw_text),
                            reasoning="".join(reasoning),
                            image_urls=images,
                            image_kinds=kinds,
                            response_id=response_id or resp.get("id", ""),
                            sources=sources,
                            search_queries=list(tool_queries),
                            conversation_id=self.conversation_id,
                            parent_response_id=user_msg_id,
                            finish_reason="stop" if status == "completed" else "length",
                        )
                        self.last_parent_response_id = result.response_id
                        if (
                            attachment_ids
                            and images
                            and not "".join(text)
                            and not "".join(reasoning)
                            and "edited" not in kinds
                        ):
                            # Degraded gateways intermittently ignore both the
                            # message and its attachments and complete the turn
                            # with a fresh placeholder generation (captured live
                            # 2026-08-31: "give this cat a hat" + a cat photo
                            # answered by a render_generated_image of an unrelated
                            # scene with zero reasoning and zero text; healthy
                            # edits stream render_edited_image). Quarantine and
                            # fail over like any other degraded turn instead of
                            # shipping an unrelated image as a successful edit.
                            raise GatewayError(
                                "degraded",
                                "image turn returned a fresh generation unrelated "
                                "to the request (degraded account)",
                            )
                        for u, k in zip(images, kinds):
                            yield {"type": "image_url", "url": u, "kind": k}
                        completed_turn = True
                        yield {
                            "type": "done",
                            "result": result,
                            "usage": resp.get("usage") or {},
                        }
                        return
            finally:
                if not completed_turn:
                    # If aborted/cancelled mid-turn, close the WebSocket so stale unread frames
                    # don't corrupt subsequent turns on this session.
                    await self.close()
                    self.last_parent_response_id = ""
