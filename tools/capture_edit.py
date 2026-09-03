"""Capture raw mgw gateway frames for an image-edit turn ("give this cat a hat").

Ground-truth tool for the degraded-image-edit investigation. For each attempt it
picks the next available account, uploads a real cat photo (no hat), runs one
gateway turn, records EVERY raw WebSocket frame to /tmp/inv/frames_<i>_<key>.jsonl,
and downloads the produced assets to /tmp/inv/cap_<i>_<j>.<ext>.

Run from the repo root:  python3 tools/capture_edit.py [n_attempts]
"""

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterable, Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, overload, override

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from websockets.asyncio.client import ClientConnection
from websockets.frames import CloseCode
from websockets.typing import Data, DataLike

import grok_gateway as gw
import server
from grok_gateway import GatewayError, GrokSession, TurnResult
from uploads import upload_file

OUT = Path("/tmp/inv")
ATTACH = OUT / "outC.jpg"  # real cat photo, no hat
PROMPT = "give this cat a hat"


class TeeWS(ClientConnection):
    """Delegating wrapper that records every raw frame to a jsonl file."""

    def __init__(self, ws: ClientConnection, path: Path) -> None:
        # Wrap a live connection; delegate protocol state instead of
        # initialising a fresh sans-I/O protocol.
        self._ws: ClientConnection = ws
        self._fh = path.open("a", encoding="utf-8")
        self.protocol = ws.protocol

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ws, name)

    @overload
    async def recv(self, decode: Literal[True]) -> str: ...

    @overload
    async def recv(self, decode: Literal[False]) -> bytes: ...

    @overload
    async def recv(self, decode: bool | None = None) -> Data: ...

    @override
    async def recv(self, decode: bool | None = None) -> Data:
        raw: Data = (
            await self._ws.recv(decode) if decode is not None else await self._ws.recv()
        )
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
        self, code: CloseCode | int = CloseCode.NORMAL_CLOSURE, reason: str = ""
    ) -> None:
        try:
            await self._ws.close(code, reason)
        finally:
            with suppress(OSError):
                self._fh.close()

    def _log(self, direction: str, raw: str) -> None:
        self._fh.write(
            json.dumps({"t": round(time.time(), 3), "dir": direction, "raw": raw})
            + "\n"
        )
        self._fh.flush()


async def run_one(
    i: int, acc_key: str | None = None, prompt: str = PROMPT, image: Path = ATTACH
) -> bool:
    await server.pool.reload_if_changed()
    if acc_key:
        acc = next((a for a in server.pool.snapshot() if a.key == acc_key), None)
        if acc is None:
            print(f"[{i}] account {acc_key} not in pool")
            return False
    else:
        acc = server.pool.acquire()
        if acc is None:
            print(f"[{i}] no account available")
            return False
    key = acc.key
    print(f"[{i}] account {acc.index} {key}", flush=True)
    sess = GrokSession(acc.cookie_header(), "")
    frames = OUT / f"frames_{i}_{key.replace(':', '_')}.jsonl"
    try:
        uid = await gw.resolve_user_id(acc.cookie_header())
    except GatewayError as e:
        print(f"[{i}] resolve_user_id failed: {e}")
        return False
    images: list[str] = []
    try:
        sess.user_id = uid
        await sess.connect()
        assert sess.ws is not None
        sess.ws = TeeWS(sess.ws, frames)
        from curl_cffi.requests import AsyncSession as AS

        await server.refresh_statsig_pair()
        async with AS(impersonate="chrome") as s:
            fm = await upload_file(
                s,
                acc.cookie_header(),
                server.statsig,
                image.name,
                image.read_bytes(),
                None,
            )
        fid = (fm or {}).get("fileMetadataId")
        print(f"[{i}] uploaded file_id={fid}", flush=True)
        if not fid:
            print(f"[{i}] upload returned no file id; skipping turn")
            return False
        async for ev in sess.ask(prompt, attachment_ids=[fid], user_text=prompt):
            t = ev.get("type")
            if t == "text_delta":
                pass
            elif t == "image_url":
                images.append(ev.get("url", ""))
                print(f"[{i}] image: {ev.get('url', '')[:110]}", flush=True)
            elif t == "done":
                res = ev.get("result")
                assert isinstance(res, TurnResult)
                assert isinstance(res.text, str)
                assert isinstance(res.reasoning, str)
                assert isinstance(res.image_urls, list)
                print(
                    f"[{i}] done: text={len(res.text)}ch "
                    f"reasoning={len(res.reasoning)}ch images={len(res.image_urls)}",
                    flush=True,
                )
                images = list(res.image_urls)
        for j, u in enumerate(images):
            try:
                data, _mime, name = await server._download_asset(u, acc.cookie_header())
                p = OUT / f"cap_{i}_{j}{Path(name).suffix or '.jpg'}"
                p.write_bytes(data)
                print(f"[{i}] saved {p.name} ({len(data)}b)", flush=True)
            except (
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
                AttributeError,
                TimeoutError,
            ) as e:
                print(f"[{i}] download failed {u[:80]}: {e}", flush=True)
        return True
    except GatewayError as e:
        print(f"[{i}] gateway error ({e.kind}): {e}", flush=True)
        return False
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        TimeoutError,
    ) as e:
        print(f"[{i}] error: {type(e).__name__}: {e}", flush=True)
        return False
    finally:
        await sess.close()


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    prompt = sys.argv[2] if len(sys.argv) > 2 else PROMPT
    image = Path(sys.argv[3]) if len(sys.argv) > 3 else ATTACH
    acc_key = sys.argv[4] if len(sys.argv) > 4 else None
    for i in range(n):
        try:
            await run_one(i, acc_key=acc_key, prompt=prompt, image=image)
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
        ) as e:
            print(f"[{i}] fatal: {type(e).__name__}: {e}", flush=True)
        await asyncio.sleep(2)
    print("capture done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
