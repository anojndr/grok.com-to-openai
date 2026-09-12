# Copyright (c) 2026 grok-to-openai-api contributors.
"""Capture raw mgw gateway frames for an image-edit turn.

Ground-truth tool for the degraded-image-edit investigation. For each attempt it
picks the next available account, uploads a real cat photo (no hat), runs one
gateway turn, records EVERY raw WebSocket frame to frames_<i>_<key>.jsonl,
and downloads the produced assets to cap_<i>_<j>.<ext>.

Run from the repo root:  python3 tools/capture_edit.py [n_attempts]
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload, override

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from curl_cffi.requests import AsyncSession
from websockets.asyncio.client import ClientConnection
from websockets.frames import CloseCode

import grok_gateway as gw
import server
from grok_gateway import GatewayError, GrokSession, TurnResult
from uploads import UploadPayload, upload_file

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Iterable

    from websockets.typing import Data, DataLike

    from accounts import Account

logger = logging.getLogger(__name__)

OUT = Path(tempfile.gettempdir()) / "inv"
ATTACH = OUT / "outC.jpg"  # real cat photo, no hat
PROMPT = "give this cat a hat"

DEFAULT_ATTEMPTS = 3
PROMPT_ARG_INDEX = 2
IMAGE_ARG_INDEX = 3
ACCOUNT_ARG_INDEX = 4
INTER_ATTEMPT_DELAY_SECONDS = 2


class TeeWS(ClientConnection):
    """Delegating wrapper that records every raw frame to a jsonl file."""

    def __init__(self, ws: ClientConnection, path: Path) -> None:
        """Wrap a live connection and open the frame log.

        Args:
            ws: Live gateway connection to delegate to.
            path: JSONL file receiving raw frames.

        """
        # Wrap a live connection; delegate protocol state instead of
        # initialising a fresh sans-I/O protocol.
        self._ws: ClientConnection = ws
        self._fh = path.open("a", encoding="utf-8")
        self.protocol = ws.protocol

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attributes to the wrapped connection.

        Args:
            name: Attribute name.

        Returns:
            Wrapped connection attribute.

        """
        return getattr(self._ws, name)

    @overload
    async def recv(self, decode: Literal[True]) -> str: ...

    @overload
    async def recv(self, decode: Literal[False]) -> bytes: ...

    @overload
    async def recv(self, decode: object = None) -> Data: ...

    @override
    async def recv(self, decode: object = None) -> Data:
        if decode is None:
            raw: Data = await self._ws.recv()
        else:
            if not isinstance(decode, bool):
                msg = f"unexpected decode flag: {type(decode).__name__}"
                raise TypeError(msg)
            raw = await self._ws.recv(decode)
        self._log("recv", raw if isinstance(raw, str) else repr(raw))
        return raw

    @override
    async def send(
        self,
        message: DataLike | Iterable[DataLike] | AsyncIterable[DataLike],
        *,
        text: bool | None = None,
    ) -> None:
        self._log("send", message if isinstance(message, str) else str(message))
        if text is not None:
            await self._ws.send(message, text=text)
        else:
            await self._ws.send(message)

    @override
    async def close(
        self,
        code: CloseCode | int = CloseCode.NORMAL_CLOSURE,
        reason: str = "",
    ) -> None:
        try:
            await self._ws.close(code, reason)
        finally:
            with suppress(OSError):
                self._fh.close()

    def _log(self, direction: str, raw: str) -> None:
        self._fh.write(
            json.dumps({"t": round(time.time(), 3), "dir": direction, "raw": raw})
            + "\n",
        )
        self._fh.flush()


async def _acquire_account(
    attempt: int,
    acc_key: str | None,
) -> Account | None:
    """Acquire the requested or next available account.

    Args:
        attempt: Attempt index used for logging.
        acc_key: Optional account key to pin, else acquire any.

    Returns:
        Account or None when unavailable.

    """
    await server.pool.reload_if_changed()
    if acc_key is not None:
        acc = next((a for a in server.pool.snapshot() if a.key == acc_key), None)
        if acc is None:
            logger.warning("[%d] account %s not in pool", attempt, acc_key)
            return None
        return acc
    acc = server.pool.acquire()
    if acc is None:
        logger.warning("[%d] no account available", attempt)
        return None
    return acc


async def _resolve_uid(attempt: int, cookie: str) -> str | None:
    """Resolve the user id for an account cookie.

    Args:
        attempt: Attempt index used for logging.
        cookie: Account cookie header.

    Returns:
        User id or None when resolution fails.

    """
    try:
        return await gw.resolve_user_id(cookie)
    except GatewayError as err:
        logger.warning("[%d] resolve_user_id failed: %s", attempt, err)
        return None


async def _ensure_connected(
    sess: GrokSession,
    uid: str,
    frames: Path,
) -> None:
    """Connect the session and wrap its socket with a frame tee.

    Args:
        sess: Gateway session to connect.
        uid: Resolved user id.
        frames: Frame log path.

    Raises:
        RuntimeError: When the session socket is missing after connect.

    """
    sess.user_id = uid
    await sess.connect()
    if sess.ws is None:
        msg = "session socket missing after connect"
        raise RuntimeError(msg)
    sess.ws = TeeWS(sess.ws, frames)


async def _upload_file_id(
    attempt: int,
    cookie: str,
    image: Path,
) -> str | None:
    """Upload the probe image and return its file id.

    Args:
        attempt: Attempt index used for logging.
        cookie: Account cookie header.
        image: Image path to upload.

    Returns:
        File metadata id or None when upload yields none.

    """
    # Mirror production: refresh the seed pair before every upload so the
    # capture signs exactly like the live flow instead of going stale.
    await server.refresh_statsig_pair()
    image_bytes = await asyncio.to_thread(image.read_bytes)
    async with AsyncSession(impersonate="chrome") as sess:
        payload = UploadPayload(
            filename=image.name,
            data=image_bytes,
            mime=None,
        )
        meta = await upload_file(sess, cookie, server.statsig, payload)
    fid_raw: object = (meta or {}).get("fileMetadataId")
    fid = fid_raw if isinstance(fid_raw, str) else None
    logger.info("[%d] uploaded file_id=%s", attempt, fid)
    if not fid:
        logger.warning("[%d] upload returned no file id; skipping turn", attempt)
        return None
    return fid


async def _collect_turn_images(
    sess: GrokSession,
    prompt: str,
    fid: str,
    attempt: int,
) -> list[str]:
    """Run one gateway turn and collect produced image URLs.

    Args:
        sess: Connected gateway session.
        prompt: Edit prompt.
        fid: Uploaded file id.
        attempt: Attempt index used for logging.

    Returns:
        Image URLs produced by the turn.

    Raises:
        TypeError: When the done event result has an unexpected shape.

    """
    images: list[str] = []
    async for event in sess.ask(prompt, attachment_ids=[fid], user_text=prompt):
        event_type = event.get("type")
        if event_type == "text_delta":
            continue
        if event_type == "image_url":
            url_raw: object = event.get("url", "")
            url = url_raw if isinstance(url_raw, str) else ""
            images.append(url)
            logger.info("[%d] image: %s", attempt, url[:110])
            continue
        if event_type != "done":
            continue
        result: object = event.get("result")
        if not isinstance(result, TurnResult):
            msg = f"unexpected done result: {type(result).__name__}"
            raise TypeError(msg)
        if not isinstance(result.text, str):
            msg = "done result text is not a string"
            raise TypeError(msg)
        if not isinstance(result.reasoning, str):
            msg = "done result reasoning is not a string"
            raise TypeError(msg)
        if not isinstance(result.image_urls, list):
            msg = "done result image_urls is not a list"
            raise TypeError(msg)
        logger.info(
            "[%d] done: text=%dch reasoning=%dch images=%d",
            attempt,
            len(result.text),
            len(result.reasoning),
            len(result.image_urls),
        )
        images = list(result.image_urls)
    return images


async def _download_images(
    attempt: int,
    images: list[str],
    cookie: str,
) -> None:
    """Download turn images to the output directory.

    Args:
        attempt: Attempt index used for logging.
        images: Image URLs to download.
        cookie: Account cookie header.

    """
    for seq, url in enumerate(images):
        try:
            data, _mime, name = await server.download_asset(url, cookie)
            dest = OUT / f"cap_{attempt}_{seq}{Path(name).suffix or '.jpg'}"
            await asyncio.to_thread(dest.write_bytes, data)
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as err:
            logger.warning("[%d] download failed %s: %s", attempt, url[:80], err)
            continue
        except TimeoutError as err:
            logger.warning("[%d] download failed %s: %s", attempt, url[:80], err)
            continue
        else:
            logger.info("[%d] saved %s (%db)", attempt, dest.name, len(data))


async def run_one(
    attempt: int,
    acc_key: str | None = None,
    prompt: str = PROMPT,
    image: Path = ATTACH,
) -> bool:
    """Capture gateway frames for one image-edit attempt.

    Args:
        attempt: Attempt index used for logging and file names.
        acc_key: Optional account key to pin.
        prompt: Edit prompt.
        image: Probe image path.

    Returns:
        True when the turn completed, False otherwise.

    """
    acc = await _acquire_account(attempt, acc_key)
    if acc is None:
        return False
    uid = await _resolve_uid(attempt, acc.cookie_header())
    if uid is None:
        return False
    key = acc.key
    logger.info("[%d] account %d %s", attempt, acc.index, key)
    sess = GrokSession(acc.cookie_header(), "")
    frames = OUT / f"frames_{attempt}_{key.replace(':', '_')}.jsonl"
    try:
        await _ensure_connected(sess, uid, frames)
        fid = await _upload_file_id(attempt, acc.cookie_header(), image)
        if not fid:
            return False
        images = await _collect_turn_images(sess, prompt, fid, attempt)
        await _download_images(attempt, images, acc.cookie_header())
    except GatewayError as err:
        logger.warning("[%d] gateway error (%s): %s", attempt, err.kind, err)
        return False
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        TimeoutError,
    ) as err:
        logger.warning("[%d] error: %s: %s", attempt, type(err).__name__, err)
        return False
    else:
        return True
    finally:
        await sess.close()


async def main() -> None:
    """Capture frames for the requested number of attempts."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ATTEMPTS
    prompt = sys.argv[PROMPT_ARG_INDEX] if len(sys.argv) > PROMPT_ARG_INDEX else PROMPT
    image_arg = sys.argv[IMAGE_ARG_INDEX] if len(sys.argv) > IMAGE_ARG_INDEX else None
    image = Path(image_arg) if image_arg is not None else ATTACH
    acc_key = sys.argv[ACCOUNT_ARG_INDEX] if len(sys.argv) > ACCOUNT_ARG_INDEX else None
    for attempt in range(n):
        try:
            await run_one(attempt, acc_key=acc_key, prompt=prompt, image=image)
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            AssertionError,
            TimeoutError,
            GatewayError,
        ) as err:
            logger.warning(
                "[%d] fatal: %s: %s",
                attempt,
                type(err).__name__,
                err,
            )
        await asyncio.sleep(INTER_ATTEMPT_DELAY_SECONDS)
    logger.info("capture done")


if __name__ == "__main__":
    asyncio.run(main())
