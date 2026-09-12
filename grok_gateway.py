# Copyright (c) 2026 grok-to-openai-api contributors.
"""Grok WebSocket gateway client for the grok.com mgw endpoint.

A GrokSession wraps one live gateway connection, which is one grok
conversation. Sequential ask calls on the same session are proper
multi-turn: grok keeps the full context server-side, so only the new
user message travels the wire.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn

import websockets
from curl_cffi.requests import AsyncSession
from websockets.protocol import State

from config import GROK_BASE, USER_AGENT

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from websockets.asyncio.client import ClientConnection

type JsonValue = (
    dict[str, JsonValue] | list[JsonValue] | str | int | float | bool | None
)

ASSET_BASE = "https://assets.grok.com/"

_IMAGE_PROGRESS_COMPLETE = 100
_MIN_UNRELATED_QUERIES = 2
_HTTP_OK = 200
_DEFAULT_IDLE_TIMEOUT = 120.0
_DEFAULT_MAX_TURN_TIMEOUT = 300.0

_TAG_START = "<grok:render"
_TAG_END = "</grok:render>"

_RESULT_LIST_KEYS = ("web_results", "webSearch_results", "webpages", "results")
_WEB_SEARCH_LIST_KEYS = ("webpages", "results", "web_results")
_TOOL_LIST_KEYS = ("web_search_results", "web_results", "webpages", "results")
_IMAGE_CARD_KEYS = (
    "render_edited_image",
    "render_generated_image",
    "imageChunk",
    "image_attachment",
    "imageAttachment",
    "image",
    "generatedImage",
    "media",
)
_TEXT_IMAGE_KEYS = ("generated_image", "image")

RENDER_TAG_RE = re.compile(r"<grok:render\b.*?(?:</grok:render>|$)", re.DOTALL)
URL_OR_PATH_RE = re.compile(
    r'(?:https?://assets\.grok\.com/users/[^\s"\'<>]+\.(?:jpg|jpeg|png|webp|gif)|users/[0-9a-fA-F-]+/(?:generated|attachments)/[^\s"\'<>]+\.(?:jpg|jpeg|png|webp|gif))',
    re.IGNORECASE,
)


def strip_render_tags(text: str) -> str:
    """Remove grok:render pseudo-HTML card markup from streamed text.

    Args:
        text: Complete turn text that may embed render cards.

    Returns:
        The text with every render tag stripped out.

    """
    return RENDER_TAG_RE.sub("", text)


def _partial_tag_len(buffer: str) -> int:
    """Measure a trailing partial render-tag prefix.

    Args:
        buffer: Buffered text whose tail may start a render tag.

    Returns:
        Length of the longest tail that prefixes a render tag.

    """
    width = min(len(_TAG_START), len(buffer) + 1)
    matched = 0
    for size in range(1, width):
        if _TAG_START.startswith(buffer[-size:]):
            matched = size
    return matched


class RenderFilter:
    """Incrementally strip render tags from streaming text deltas.

    Text outside tags passes through; tag bytes are buffered until the
    closing tag arrives, so cards never leak into streamed output.
    """

    def __init__(self) -> None:
        """Hold an empty buffer outside any render tag."""
        self._buffer: str = ""
        self._in_tag: bool = False

    def _emit_outside_tag(self, out: list[str]) -> bool:
        """Emit text up to the next render tag.

        Args:
            out: Sink for text that is safe to stream.

        Returns:
            True when scanning should continue with the updated buffer.

        """
        idx = self._buffer.find(_TAG_START)
        if idx != -1:
            if idx > 0:
                out.append(self._buffer[:idx])
                self._buffer = self._buffer[idx:]
            self._in_tag = True
            return True
        partial_len = _partial_tag_len(self._buffer)
        if partial_len > 0:
            safe = self._buffer[:-partial_len]
            self._buffer = self._buffer[-partial_len:]
            if safe:
                out.append(safe)
            return False
        out.append(self._buffer)
        self._buffer = ""
        return False

    def _skip_inside_tag(self) -> bool:
        """Skip past a closing render tag when one is buffered.

        Returns:
            True when scanning should continue with the updated buffer.

        """
        close_idx = self._buffer.find(_TAG_END)
        if close_idx == -1:
            return False
        self._buffer = self._buffer[close_idx + len(_TAG_END) :]
        self._in_tag = False
        return True

    def process(self, chunk: str) -> str:
        """Filter a streaming text delta through the render-tag filter.

        Args:
            chunk: Raw text delta from the gateway.

        Returns:
            The delta with render-tag bytes removed.

        """
        if not chunk:
            return ""
        self._buffer += chunk
        out: list[str] = []
        while self._buffer:
            if self._in_tag:
                if not self._skip_inside_tag():
                    break
            elif not self._emit_outside_tag(out):
                break
        return "".join(out)

    def flush(self) -> str:
        """Emit buffered text left over at the end of a turn.

        Returns:
            Remaining text when no render tag is open, else empty.

        """
        # At the end of a turn, flush only when not inside a tag.
        if not self._in_tag:
            res = self._buffer
            self._buffer = ""
            return res
        self._buffer = ""
        return ""


def _asset_url(url: str) -> str:
    """Expand a gateway asset path into an absolute URL.

    Args:
        url: Asset path or already-absolute URL.

    Returns:
        The absolute URL for the asset.

    """
    if url.startswith(("data:", "http://", "https://")):
        return url
    return ASSET_BASE + url.lstrip("/")


def _is_complete_progress(progress: JsonValue) -> bool:
    """Report whether an image-chunk progress value means finished.

    Args:
        progress: Raw progress value from an image chunk.

    Returns:
        True when the value is numeric and at least complete.

    """
    return isinstance(progress, (int, float)) and (progress >= _IMAGE_PROGRESS_COMPLETE)


def _scan_value_for_asset(raw: JsonValue) -> str | None:
    """Search a payload value for an embedded grok asset path.

    Args:
        raw: Single payload value to inspect.

    Returns:
        The matched asset URL, or None when there is no match.

    """
    if not isinstance(raw, str):
        return None
    if "assets.grok.com/users/" not in raw and not raw.startswith("users/"):
        return None
    match = URL_OR_PATH_RE.search(raw)
    if match is None:
        return None
    return _asset_url(match.group(0))


def chunk_image_url(value: JsonValue) -> str | None:
    """Extract a finished image URL or data URI from a chunk payload.

    Image generation and editing arrive as card chunks holding an inner
    image chunk with a progress marker, where mid-progress carries a data
    URI placeholder and completion carries the final asset path. Legacy
    flat shapes with direct URL fields are tolerated as well.

    Args:
        value: Gateway chunk payload to inspect.

    Returns:
        The finished image URL, or None for placeholders and non-images.

    """
    if not isinstance(value, dict):
        return None
    image_chunk = value.get("image_chunk")
    if isinstance(image_chunk, dict):
        url = image_chunk.get("imageUrl") or image_chunk.get("image_url") or ""
        progress = image_chunk.get("progress")
        if (
            isinstance(url, str)
            and url
            and (progress is None or _is_complete_progress(progress))
        ):
            return _asset_url(url)
        return None
    url = value.get("imageUrl") or value.get("url") or value.get("image_url")
    if isinstance(url, str) and url:
        return _asset_url(url)
    # Card payloads sometimes carry the asset path in an unexpected field;
    # scan the values for a grok asset path before giving up.
    for raw in value.values():
        found = _scan_value_for_asset(raw)
        if found is not None:
            return found
    return None


def _card_query(card: JsonValue) -> str | None:
    """Extract the search query from a tool usage card payload.

    Live frames carry the query on the web search args, with flat and
    output-placed variants tolerated across gateway builds.

    Args:
        card: Tool usage card payload to inspect.

    Returns:
        The stripped query, or None when no usable query is present.

    """
    if not isinstance(card, dict):
        return None
    search = card.get("web_search")
    if not isinstance(search, dict):
        return None
    args = search.get("args")
    query = args.get("query") if isinstance(args, dict) else None
    if not isinstance(query, str):
        query = search.get("query")
    if not isinstance(query, str) or not query:
        return None
    stripped = query.strip()
    if not stripped:
        return None
    return stripped


def _append_result_entry(
    results: list[dict[str, Any]],
    seen: set[str],
    item: JsonValue,
) -> None:
    """Append one citation entry unless its URL was already recorded.

    Args:
        results: Sink for normalized citation entries.
        seen: Lowercased URLs already present in results.
        item: Raw citation candidate to normalize.

    """
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
    results.append({"url": url, "title": title})


def _collect_entries(
    results: list[dict[str, Any]],
    seen: set[str],
    mapping: dict[str, JsonValue],
    keys: tuple[str, ...],
) -> None:
    """Collect citation lists stored under any of the given keys.

    Args:
        results: Sink for normalized citation entries.
        seen: Lowercased URLs already present in results.
        mapping: Payload that may hold citation lists.
        keys: Candidate keys holding lists of citation entries.

    """
    for key in keys:
        items = mapping.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            _append_result_entry(results, seen, item)


def _scan_container(
    results: list[dict[str, Any]],
    seen: set[str],
    container: JsonValue,
) -> None:
    """Scan one payload container for every known citation shape.

    Args:
        results: Sink for normalized citation entries.
        seen: Lowercased URLs already present in results.
        container: Payload container to scan for citations.

    """
    if not isinstance(container, dict):
        return
    card_attachment = container.get("card_attachment")
    if isinstance(card_attachment, dict):
        _append_result_entry(results, seen, card_attachment)
    result = container.get("result")
    if isinstance(result, dict):
        _collect_entries(results, seen, result, _RESULT_LIST_KEYS)
    tool_result = container.get("tool_result")
    if not isinstance(tool_result, dict):
        return
    web_search = tool_result.get("web_search")
    if isinstance(web_search, dict):
        _collect_entries(results, seen, web_search, _WEB_SEARCH_LIST_KEYS)
    _collect_entries(results, seen, tool_result, _TOOL_LIST_KEYS)


def extract_web_results(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize citation entries out of a gateway event.

    Multiple event shapes carry web citations: chunk tool results, grok
    output tool results, card attachments, and search result payloads.

    Args:
        event: Decoded gateway frame to scan for citations.

    Returns:
        Deduplicated citation entries with url and title keys.

    """
    if not isinstance(event, dict):
        return []
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_nested = event.get("event")
    nested = raw_nested if isinstance(raw_nested, dict) else {}
    for container in (event, nested):
        if not isinstance(container, dict):
            continue
        _scan_container(results, seen, container)
        _scan_container(results, seen, container.get("output"))
        _scan_container(results, seen, container.get("chunk"))
    return results


class GatewayError(Exception):
    """Gateway failure carrying a machine-readable kind label."""

    def __init__(self, kind: str, message: str) -> None:
        """Store the failure kind alongside the message.

        Args:
            kind: Machine-readable failure category.
            message: Human-readable failure description.

        """
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
    },
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
    },
)


def unrelated_queries(queries: list[str], user_text: str) -> bool:
    """Report whether tool search queries look unrelated to the request.

    A known placeholder query fails immediately on its own, even when the
    prompt is short or empty. Otherwise two or more unrelated queries
    signal degraded behavior; duplicated identical queries count toward
    that total, while a single unrelated query is tolerated as paraphrase.

    Args:
        queries: Web-search queries observed on the turn.
        user_text: Raw user text the queries should relate to.

    Returns:
        True when the queries show the degraded-search signature.

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
    # Two or more unrelated queries signal degraded behavior; repeated
    # identical unrelated queries also count, since the gateway is stuck
    # searching the same off-topic content.
    if len(set(unrelated)) >= _MIN_UNRELATED_QUERIES:
        return True
    return len(unrelated) >= _MIN_UNRELATED_QUERIES


@dataclass
class TurnResult:
    """Final result of one completed gateway turn.

    Attributes:
        text: Answer text with render markup stripped.
        reasoning: Side-channel thinking text streamed during the turn.
        image_urls: Image URLs surfaced during the turn, first-seen order.
        image_kinds: Parallel to image_urls: edited, generated, or unknown.
        sources: Normalized web citation entries.
        search_queries: Web-search queries observed during the turn.
        response_id: Gateway id of the assistant response.
        conversation_id: Gateway conversation the turn belongs to.
        parent_response_id: User message id this turn answers.
        finish_reason: Stop for clean completion, length for truncation.

    """

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


@dataclass
class _TurnState:
    """Mutable per-turn accumulator for one streaming gateway turn.

    Attributes:
        text: Answer text deltas in arrival order.
        reasoning: Thinking deltas in arrival order.
        images: Image URLs surfaced so far, first-seen order.
        kinds: Parallel to images: edited, generated, or unknown.
        tool_queries: Web-search queries observed so far.
        sources: Normalized web citation entries.
        seen_source_urls: Lowercased source URLs already recorded.
        response_id: Gateway id of the assistant response, when known.
        user_message_id: Gateway id of the user message, when known.
        started_at: Monotonic start time used for the turn budget.
        last_activity_at: Last time a frame arrived, for idle detection.
        attachment_ids: File ids mentioned on this turn, if any.

    """

    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    tool_queries: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    seen_source_urls: set[str] = field(default_factory=set)
    response_id: str = ""
    user_message_id: str = ""
    started_at: float = 0.0
    last_activity_at: float = 0.0
    attachment_ids: list[str] | None = None


def new_uuid() -> str:
    """Create a random hex id for gateway event envelopes.

    Returns:
        A new random UUID string.

    """
    return str(uuid.uuid4())


def default_x_grok() -> dict[str, Any]:
    """Build the default extended session options for the gateway.

    Returns:
        The x_grok payload enabling chunks, images, and memory hygiene.

    """
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


def _extract_user_id(data: JsonValue) -> str:
    """Pick the user id out of a decoded auth-session payload.

    Args:
        data: Decoded JSON body from the auth session endpoint.

    Returns:
        The user id, or an empty string when none is present.

    """
    if not isinstance(data, dict):
        return ""
    user = data.get("user")
    user_id = user.get("id") if isinstance(user, dict) else None
    session = data.get("session")
    session_user_id = session.get("userId") if isinstance(session, dict) else None
    raw = user_id or session_user_id or data.get("userId") or ""
    return raw if isinstance(raw, str) else ""


async def resolve_user_id(cookie_header: str) -> str:
    """Resolve the grok user id for a cookie header.

    Args:
        cookie_header: Raw Cookie header value for grok.com.

    Returns:
        The authenticated grok user id.

    Raises:
        GatewayError: When the lookup fails or yields no user id.

    """
    async with AsyncSession[Any](impersonate="chrome") as session:
        response = await session.get(
            f"{GROK_BASE}/api/auth/session",
            headers={"user-agent": USER_AGENT, "cookie": cookie_header},
            timeout=10,
        )
        if response.status_code != _HTTP_OK:
            kind = "auth"
            msg = f"failed to resolve user id ({response.status_code})"
            raise GatewayError(kind, msg)
        try:
            data = response.json()
        except (ValueError, TypeError, AttributeError):
            data = None
        user_id = _extract_user_id(data)
        if user_id:
            return user_id
        kind = "auth"
        msg = f"failed to resolve user id ({response.status_code})"
        raise GatewayError(kind, msg)


def _session_headers(cookie_header: str) -> dict[str, str]:
    """Build the WebSocket handshake headers for the gateway.

    Args:
        cookie_header: Raw Cookie header value for grok.com.

    Returns:
        Headers for the mgw WebSocket connection.

    """
    return {
        "Origin": GROK_BASE,
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Cookie": cookie_header,
    }


async def _send_session_create(
    connection: ClientConnection,
    conversation_id: str,
    model_mode: str,
) -> None:
    """Send the session.create opener on a fresh gateway socket.

    Args:
        connection: Fresh gateway WebSocket connection.
        conversation_id: Conversation to resume, if any.
        model_mode: Model mode negotiated for this session.

    """
    xgrok = default_x_grok()
    if conversation_id:
        xgrok["conversation_id"] = conversation_id
    await connection.send(
        json.dumps(
            {
                "event": {
                    "type": "session.create",
                    "event_id": "evt_init_" + new_uuid(),
                    "session": {"model": model_mode, "x_grok": xgrok},
                },
            },
        ),
    )


def _parse_handshake_frame(raw: str | bytes) -> dict[str, Any]:
    """Decode one handshake frame from the gateway.

    Args:
        raw: Raw WebSocket text frame.

    Returns:
        The decoded frame envelope.

    Raises:
        GatewayError: When the frame is not valid JSON.

    """
    try:
        return json.loads(raw)
    except (ValueError, TypeError, AttributeError) as err:
        kind = "upstream"
        msg = f"malformed json in handshake: {err}"
        raise GatewayError(kind, msg) from err


def _build_turn_chunks(
    prompt: str,
    attachment_ids: list[str] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble ordered input chunks for a user message.

    Args:
        prompt: User message text for this turn.
        attachment_ids: File ids to mention on this turn.
        system_prompt: Leading system text prepended to the turn.

    Returns:
        Input chunks in gateway order: system, mentions, then text.

    """
    chunks: list[dict[str, Any]] = []
    if system_prompt:
        chunks.append({"text": {"text": system_prompt + "\n\n"}})
    chunks.extend(
        {"mention": {"file_mention": {"file_id": attachment_id}}}
        for attachment_id in attachment_ids or []
    )
    chunks.append({"text": {"text": prompt}})
    return chunks


async def _send_turn_openers(
    connection: ClientConnection,
    conversation_id: str,
    parent_response_id: str,
    chunks: list[dict[str, Any]],
) -> None:
    """Queue the user message and arm the assistant response.

    Args:
        connection: Live gateway WebSocket connection.
        conversation_id: Active gateway conversation id.
        parent_response_id: Parent response anchoring the turn, if any.
        chunks: Ordered input chunks for the user message.

    """
    now_ms = int(time.time() * 1000)
    item: dict[str, Any] = {
        "type": "message",
        "role": "user",
        "x_grok": {
            "client_message_id": new_uuid(),
            "input_chunks": chunks,
        },
    }
    item_event: dict[str, Any] = {
        "session_id": conversation_id,
        "event": {
            "type": "conversation.item.create",
            "event_id": f"evt_msg_{now_ms}_{new_uuid()[:8]}",
            "item": item,
        },
    }
    if parent_response_id:
        item_event["event"]["parent_response_id"] = parent_response_id
    await connection.send(json.dumps(item_event))
    await connection.send(
        json.dumps(
            {
                "session_id": conversation_id,
                "event": {
                    "type": "response.create",
                    "event_id": f"evt_resp_{now_ms}_{new_uuid()[:8]}",
                },
            },
        ),
    )


def _ensure_turn_budget(state: _TurnState, max_turn_timeout: float) -> None:
    """Fail the turn when it exceeds its total duration budget.

    Args:
        state: Mutable per-turn accumulator.
        max_turn_timeout: Maximum turn duration in seconds.

    Raises:
        GatewayError: When the turn ran longer than the budget.

    """
    if time.time() - state.started_at > max_turn_timeout:
        kind = "timeout"
        msg = "gateway turn exceeded max duration"
        raise GatewayError(kind, msg)


async def _receive_turn_frame(
    connection: ClientConnection,
    state: _TurnState,
    idle_timeout: float,
) -> str | bytes:
    """Wait for the next gateway frame of a turn.

    Args:
        connection: Live gateway WebSocket connection.
        state: Mutable per-turn accumulator.
        idle_timeout: Maximum idle seconds between frames.

    Returns:
        The raw WebSocket frame.

    Raises:
        GatewayError: On idle timeout or a closed connection.

    """
    try:
        raw = await asyncio.wait_for(
            connection.recv(),
            timeout=max(1.0, idle_timeout - (time.time() - state.last_activity_at)),
        )
    except TimeoutError as err:
        kind = "timeout"
        msg = "gateway idle timeout"
        raise GatewayError(kind, msg) from err
    except websockets.ConnectionClosed as err:
        kind = "closed"
        msg = "connection closed mid-turn"
        raise GatewayError(kind, msg) from err
    else:
        state.last_activity_at = time.time()
        return raw


def _parse_turn_frame(raw: str | bytes) -> dict[str, Any] | None:
    """Decode one turn frame, skipping malformed payloads.

    Args:
        raw: Raw WebSocket frame.

    Returns:
        The decoded frame envelope, or None when it is malformed.

    """
    try:
        return json.loads(raw)
    except (ValueError, TypeError, AttributeError) as err:
        logging.getLogger("uvicorn.error").debug(
            "skipping malformed gateway frame: %s",
            err,
        )
        return None


def _raise_for_error_event(event: dict[str, Any]) -> NoReturn:
    """Raise the GatewayError described by an error event.

    Args:
        event: Inner gateway event payload.

    Raises:
        GatewayError: Always, with the upstream error message.

    """
    error = event.get("error") or {}
    msg = error.get("message") or json.dumps(error)[:200]
    kind = "upstream"
    raise GatewayError(kind, msg)


def _raise_for_stream_error(stream_error: dict[str, Any]) -> NoReturn:
    """Raise the GatewayError described by a stream error payload.

    Args:
        stream_error: Stream error payload from grok output.

    Raises:
        GatewayError: Quota for usage limits, else upstream.

    """
    kind_value = stream_error.get("kind", "")
    message = stream_error.get("message", "stream error")
    kind = "quota" if "usage_limit" in kind_value else "upstream"
    raise GatewayError(kind, message)


def _collect_sources(state: _TurnState, env: dict[str, Any]) -> None:
    """Record unseen web citations from a gateway frame.

    Args:
        state: Mutable per-turn accumulator.
        env: Decoded gateway frame.

    """
    for source in extract_web_results(env):
        key = source["url"].lower()
        if key not in state.seen_source_urls:
            state.seen_source_urls.add(key)
            state.sources.append(source)


def _register_tool_query(
    state: _TurnState,
    query: str | None,
    user_text: str,
) -> None:
    """Track a tool search query and fail degraded turns.

    Args:
        state: Mutable per-turn accumulator.
        query: Search query from the current card, if any.
        user_text: Raw user text used for degraded-search detection.

    Raises:
        GatewayError: When the queries look unrelated to the request.

    """
    if not query:
        return
    state.tool_queries.append(query)
    if unrelated_queries(state.tool_queries, user_text):
        kind = "degraded"
        msg = "gateway searched content unrelated to the request (degraded account)"
        raise GatewayError(kind, msg)


def _collect_chunk_images(
    chunk: dict[str, Any],
    state: _TurnState,
) -> None:
    """Accumulate finished image URLs from a response chunk.

    Args:
        chunk: Chunk payload that may carry image cards.
        state: Mutable per-turn accumulator.

    """
    for key in _IMAGE_CARD_KEYS:
        url = chunk_image_url(chunk.get(key))
        if url and url not in state.images:
            state.images.append(url)
            state.kinds.append(
                "edited" if key == "render_edited_image" else "generated",
            )


def _collect_text_images(raw_text: str, state: _TurnState) -> None:
    """Accumulate asset URLs embedded in the final answer text.

    Args:
        raw_text: Joined answer text deltas.
        state: Mutable per-turn accumulator.

    """
    for match in URL_OR_PATH_RE.findall(raw_text):
        image_url = _asset_url(match)
        if image_url not in state.images:
            state.images.append(image_url)
            state.kinds.append("unknown")


def _apply_search_result(
    env: dict[str, Any],
    state: _TurnState,
) -> tuple[list[dict[str, Any]], bool]:
    """Record web citations from a search-result event.

    Args:
        env: Decoded gateway frame.
        state: Mutable per-turn accumulator.

    Returns:
        An empty event list with the turn still open.

    """
    _collect_sources(state, env)
    return [], False


def _apply_grok_output(
    event: dict[str, Any],
    env: dict[str, Any],
    state: _TurnState,
    user_text: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Apply a grok output event to the turn state.

    Args:
        event: Inner gateway event payload.
        env: Decoded gateway frame.
        state: Mutable per-turn accumulator.
        user_text: Raw user text used for degraded-search detection.

    Returns:
        An empty event list with the turn still open.

    """
    output = event.get("output") or {}
    stream_error = output.get("stream_error")
    if stream_error:
        _raise_for_stream_error(stream_error)
    _register_tool_query(
        state,
        _card_query(output.get("tool_usage_card")),
        user_text,
    )
    _collect_sources(state, env)
    for obj_key in _TEXT_IMAGE_KEYS:
        url = chunk_image_url(output.get(obj_key))
        if url and url not in state.images:
            state.images.append(url)
            state.kinds.append("generated")
    return [], False


def _apply_chunk_event(
    event: dict[str, Any],
    env: dict[str, Any],
    state: _TurnState,
    user_text: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Apply a response chunk event to the turn state.

    Args:
        event: Inner gateway event payload.
        env: Decoded gateway frame.
        state: Mutable per-turn accumulator.
        user_text: Raw user text used for degraded-search detection.

    Returns:
        Delta events for the chunk with the turn still open.

    """
    chunk = event.get("chunk") or {}
    _register_tool_query(
        state,
        _card_query(chunk.get("tool_usage_card")),
        user_text,
    )
    _collect_sources(state, env)
    _collect_chunk_images(chunk, state)
    info = chunk.get("text") or {}
    value = info.get("text", "")
    channel = info.get("channel", "")
    if not value:
        return [], False
    if "NOTETAKER_HEADER" in channel:
        # Timeline title chrome ("Thinking about your request",
        # "Writing a ... story"): neither answer text nor reasoning. Live
        # healthy fast turns stream it, so it must not pollute reasoning
        # nor trip any degraded detector.
        return [], False
    if "THINKING" in channel or "NOTETAKER" in channel:
        # Thinking/summary side-channels; bare NOTETAKER is the legacy
        # wire name kept for old captures.
        state.reasoning.append(value)
        return [{"type": "reasoning_delta", "text": value}], False
    state.text.append(value)
    return [{"type": "text_delta", "text": value}], False


def _apply_delta_event(
    event_type: str,
    event: dict[str, Any],
    state: _TurnState,
) -> tuple[list[dict[str, Any]], bool]:
    """Apply an output text or reasoning delta event.

    Args:
        event_type: Gateway event type selecting the channel.
        event: Inner gateway event payload.
        state: Mutable per-turn accumulator.

    Returns:
        The delta event, or nothing when the delta is empty.

    """
    delta = event.get("delta")
    if not delta:
        return [], False
    if event_type == "response.reasoning_text.delta":
        state.reasoning.append(delta)
        return [{"type": "reasoning_delta", "text": delta}], False
    state.text.append(delta)
    return [{"type": "text_delta", "text": delta}], False


def _apply_item_event(
    event_type: str,
    event: dict[str, Any],
    state: _TurnState,
) -> None:
    """Record conversation and response ids from item events.

    Args:
        event_type: Gateway event type selecting the item kind.
        event: Inner gateway event payload.
        state: Mutable per-turn accumulator.

    """
    if event_type not in {
        "conversation.item.added",
        "response.output_item.added",
    }:
        return
    item = event.get("item") or {}
    if event_type == "conversation.item.added":
        if item.get("role") == "user":
            state.user_message_id = item.get("id") or state.user_message_id
    elif item.get("role") == "assistant":
        state.response_id = item.get("id") or state.response_id


class GrokSession:
    """Live gateway connection backing one grok conversation.

    Sequential ask calls share server-side context, so each turn sends
    only its own message. Turn timeouts are configured per session, and
    a missing or stale socket reconnects lazily on the next turn.
    """

    def __init__(
        self,
        cookie_header: str,
        user_id: str,
        model_mode: str = "fast",
        *,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        max_turn_timeout: float = _DEFAULT_MAX_TURN_TIMEOUT,
    ) -> None:
        """Attach session configuration without connecting.

        Args:
            cookie_header: Raw Cookie header value for grok.com.
            user_id: Authenticated grok user id owning the socket.
            model_mode: Model mode negotiated when connecting.
            idle_timeout: Maximum idle seconds between turn frames.
            max_turn_timeout: Maximum total seconds for one turn.

        """
        self.cookie_header, self.user_id, self.model_mode = (
            cookie_header,
            user_id,
            model_mode,
        )
        self.idle_timeout = idle_timeout
        self.max_turn_timeout = max_turn_timeout
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
        """Create a disconnected session at this conversation checkpoint.

        Args:
            model_mode: Model mode for the clone, if different.

        Returns:
            A fresh session sharing the checkpoint and timeouts.

        """
        sess = GrokSession(
            self.cookie_header,
            self.user_id,
            model_mode or self.model_mode,
            idle_timeout=self.idle_timeout,
            max_turn_timeout=self.max_turn_timeout,
        )
        sess.conversation_id = self.conversation_id
        sess.last_parent_response_id = self.last_parent_response_id
        sess.attachments = [dict(a) for a in self.attachments]
        return sess

    async def _establish(
        self,
        uri: str,
        headers: dict[str, str],
    ) -> None:
        """Open the socket and run the session handshake.

        Args:
            uri: Gateway WebSocket URI for this user.
            headers: Handshake headers for the connection.

        Raises:
            GatewayError: When the handshake reports an error.

        """
        self.ws = await websockets.connect(
            uri,
            additional_headers=headers,
            max_size=32 * 1024 * 1024,
            open_timeout=10,
            close_timeout=5,
        )
        ws = self.ws
        if ws is None:
            kind = "upstream"
            msg = "connect failed: no connection established"
            raise GatewayError(kind, msg)
        await _send_session_create(ws, self.conversation_id, self.model_mode)
        await self._await_handshake(ws)

    async def _await_handshake(self, connection: ClientConnection) -> None:
        """Poll handshake frames until the session is created.

        Args:
            connection: Fresh gateway WebSocket connection.

        Raises:
            GatewayError: When the gateway reports a handshake error.

        """
        while True:
            raw = await asyncio.wait_for(connection.recv(), timeout=15)
            env = _parse_handshake_frame(raw)
            event = env.get("event") or {}
            event_type = event.get("type")
            if event_type == "session.created":
                if not self.conversation_id:
                    self.conversation_id = env.get("session_id") or ""
                self.ws_mode = self.model_mode
                return
            if event_type == "error":
                kind = "upstream"
                msg = json.dumps(event.get("error"))[:200]
                raise GatewayError(kind, msg)

    async def connect(self) -> None:
        """Connect the session socket and negotiate the model mode.

        A reconnect never orphans the previous live connection: the old
        socket closes before the new one opens.

        Raises:
            GatewayError: When connecting or the handshake fails.

        """
        # A reconnect (e.g. model-mode switch on a warm socket) must not
        # orphan the previous live connection.
        await self.close()
        uri = f"wss://grok.com/ws/mgw/?uid={self.user_id}"
        headers = _session_headers(self.cookie_header)
        try:
            await self._establish(uri, headers)
        except GatewayError:
            await self.close()
            raise
        except TimeoutError as err:
            await self.close()
            kind = "timeout"
            msg = "gateway connect or handshake timed out"
            raise GatewayError(kind, msg) from err
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as err:
            await self.close()
            # Raw socket/TLS errors must fail over like any other retryable
            # gateway failure instead of escaping as a 500.
            kind = "upstream"
            msg = f"connect failed: {type(err).__name__}: {err}"
            raise GatewayError(kind, msg) from err

    async def close(self) -> None:
        """Close the live WebSocket, if one is attached."""
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
        """Report whether the gateway socket is currently open.

        Returns:
            True when a live open WebSocket is attached.

        """
        ws = self.ws
        if ws is None:
            return False
        try:
            state = getattr(getattr(ws, "protocol", ws), "state", None)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False
        else:
            if not isinstance(state, State):
                return False
            return state is State.OPEN

    async def _ensure_connected(self) -> ClientConnection:
        """Return a live socket, reconnecting when stale or missing.

        Returns:
            The active gateway WebSocket connection.

        Raises:
            GatewayError: When no connection is available for the turn.

        """
        if not self.alive() or self.ws_mode != self.model_mode:
            await self.connect()
        ws = self.ws
        if ws is None:
            kind = "closed"
            msg = "connection unavailable for turn"
            raise GatewayError(kind, msg)
        return ws

    def _apply_turn_event(
        self,
        env: dict[str, Any],
        state: _TurnState,
        user_text: str,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Apply one decoded frame to the turn state.

        Args:
            env: Decoded gateway frame.
            state: Mutable per-turn accumulator.
            user_text: Raw user text used for degraded-search detection.

        Returns:
            Emitted events plus whether the turn is finished.

        """
        event = env.get("event") or {}
        event_type = event.get("type", "")
        if event_type == "error":
            _raise_for_error_event(event)
        if event_type == "response.search.result":
            return _apply_search_result(env, state)
        if event_type == "response.grok.output":
            return _apply_grok_output(event, env, state, user_text)
        if event_type == "response.chunk":
            return _apply_chunk_event(event, env, state, user_text)
        if event_type in {
            "response.output_text.delta",
            "response.reasoning_text.delta",
        }:
            return _apply_delta_event(event_type, event, state)
        if event_type == "response.done":
            return self._apply_done_event(event, state)
        _apply_item_event(event_type, event, state)
        return [], False

    def _apply_done_event(
        self,
        event: dict[str, Any],
        state: _TurnState,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Finalize the turn from a response.done event.

        Args:
            event: Inner gateway event payload.
            state: Mutable per-turn accumulator.

        Returns:
            Image URL events plus the done event, with the turn finished.

        Raises:
            GatewayError: When an image turn degrades to a fresh render.

        """
        response = event.get("response") or {}
        status = response.get("status", "completed")
        raw_text = "".join(state.text)
        _collect_text_images(raw_text, state)
        result = TurnResult(
            text=strip_render_tags(raw_text),
            reasoning="".join(state.reasoning),
            image_urls=state.images,
            image_kinds=state.kinds,
            response_id=state.response_id or response.get("id", ""),
            sources=state.sources,
            search_queries=list(state.tool_queries),
            conversation_id=self.conversation_id,
            parent_response_id=state.user_message_id,
            finish_reason="stop" if status == "completed" else "length",
        )
        self.last_parent_response_id = result.response_id
        if (
            state.attachment_ids
            and state.images
            and not "".join(state.text)
            and not "".join(state.reasoning)
            and "edited" not in state.kinds
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
            kind = "degraded"
            msg = (
                "image turn returned a fresh generation unrelated "
                "to the request (degraded account)"
            )
            raise GatewayError(kind, msg)
        events: list[dict[str, Any]] = [
            {"type": "image_url", "url": url, "kind": kind}
            for url, kind in zip(state.images, state.kinds, strict=False)
        ]
        events.append(
            {
                "type": "done",
                "result": result,
                "usage": response.get("usage") or {},
            },
        )
        return events, True

    async def _stream_turn(
        self,
        prompt: str,
        *,
        attachment_ids: list[str] | None = None,
        system_prompt: str | None = None,
        user_text: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one turn and yield gateway events as they arrive.

        Args:
            prompt: User message text for this turn.
            attachment_ids: File ids to mention on this turn.
            system_prompt: Leading system text prepended to the turn.
            user_text: Raw user text used for degraded-search detection.

        Yields:
            Gateway event dicts (deltas, image urls, and done).

        """
        connection = await self._ensure_connected()
        chunks = _build_turn_chunks(
            prompt,
            attachment_ids=attachment_ids,
            system_prompt=system_prompt,
        )
        await _send_turn_openers(
            connection,
            self.conversation_id,
            self.last_parent_response_id,
            chunks,
        )
        now = time.time()
        state = _TurnState(
            started_at=now,
            last_activity_at=now,
            attachment_ids=attachment_ids,
        )
        while True:
            _ensure_turn_budget(state, self.max_turn_timeout)
            raw = await _receive_turn_frame(
                connection,
                state,
                self.idle_timeout,
            )
            env = _parse_turn_frame(raw)
            if env is None:
                continue
            events, finished = self._apply_turn_event(env, state, user_text)
            for event in events:
                yield event
            if finished:
                return

    async def ask(
        self,
        prompt: str,
        *,
        attachment_ids: list[str] | None = None,
        system_prompt: str | None = None,
        user_text: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream one user turn over the gateway connection.

        The session lock is held for the whole turn so concurrent callers
        stay sequential. It is acquired and released explicitly because an
        async generator must not hold a context manager across yields. Turn
        timeouts come from the session configuration.

        Args:
            prompt: User message text for this turn.
            attachment_ids: File ids to mention on this turn.
            system_prompt: Leading system text prepended to the turn.
            user_text: Raw user text used for degraded-search detection.

        Yields:
            Gateway event dicts (deltas, image urls, and done).

        """
        await self.lock.acquire()
        completed_turn = False
        try:
            async for event in self._stream_turn(
                prompt,
                attachment_ids=attachment_ids,
                system_prompt=system_prompt,
                user_text=user_text,
            ):
                if event.get("type") in {"image_url", "done"}:
                    completed_turn = True
                yield event
        finally:
            try:
                if not completed_turn:
                    # Aborted mid-turn: close the socket so stale unread
                    # frames cannot corrupt later turns on this session.
                    await self.close()
                    self.last_parent_response_id = ""
            finally:
                self.lock.release()
