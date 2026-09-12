# Copyright (c) 2026 grok-to-openai-api contributors.
"""Regression tests: attachments must stay visible on follow-up turns.

2026-08-28 incident: turn 1 asked "which is the best for listening to music?"
with two images attached and answered fine; turn 2 replied "top 3" and the
model answered "I'm sorry, but I can't see the images you attached." The
gateway only renders files mentioned on the CURRENT message, and the proxy
kept no memory of turn-1 uploads, so the follow-up shipped a bare prompt
(replayed signed URLs that had expired were additionally swallowed silently
by extract_attachments). Fix: uploaded file ids are remembered on the
session, re-mentioned on later turns, and replayed bytes are deduplicated
by content hash instead of re-uploaded.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, override
from unittest.mock import AsyncMock, patch

import pytest

import server
from grok_gateway import GatewayError, GrokSession
from session_store import SqliteStore
from uploads import UploadError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_RETRY_CALL_COUNT = 2


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeSess(GrokSession):
    """Records ask() calls; scripted behaviors per call.

    Script items: None -> normal done; an Exception -> raise it;
    "yield-then-raise" -> emit one delta, then raise a file error.
    """

    def __init__(
        self,
        attachments: list[dict[str, str | None]] | None = None,
        script: list[GatewayError | str | None] | None = None,
    ) -> None:
        """Initialise fake session."""
        super().__init__("", "", "fast")
        self.cookie_header = "ck"
        self.attachments = list(attachments or [])
        self.calls: list[dict[str, str | list[str] | None]] = []
        self._script: list[GatewayError | str | None] = list(script or [])

    @override
    async def ask(
        self,
        prompt: str,
        *,
        attachment_ids: list[str] | None = None,
        system_prompt: str | None = None,
        user_text: str = "",
        idle_timeout: float = 120.0,
        max_turn_timeout: float = 300.0,
    ) -> AsyncIterator[dict[str, object]]:
        self.calls.append({"prompt": prompt, "attachment_ids": attachment_ids})
        behavior: GatewayError | str | None = (
            self._script.pop(0) if self._script else None
        )
        if behavior == "yield-then-raise":
            yield {"type": "text_delta", "text": "partial"}
            kind = "upstream"
            msg = "FileAttachment not found: fid1"
            raise GatewayError(kind, msg)
        if isinstance(behavior, Exception):
            raise behavior
        yield {"type": "done", "result": SimpleNamespace(text="ok")}


async def _run_turn(
    sess: FakeSess,
    prompt: str = "top 3",
    file_jobs: list[dict[str, bytes | str]] | None = None,
) -> list[dict[str, object]]:
    if file_jobs is None:
        return [ev async for ev in server.stream_session_turn(sess, prompt)]
    return [
        ev
        async for ev in server.stream_session_turn(
            sess,
            prompt,
            file_jobs=file_jobs,
        )
    ]


class ReMentionTests(unittest.IsolatedAsyncioTestCase):
    """Cover re-mention of remembered images on follow-up turns."""

    @staticmethod
    async def test_followup_turn_rementions_prior_images() -> None:
        """The incident: a follow-up with no attachments re-mentions turn-1 files."""
        sess = FakeSess(
            attachments=[
                {"file_id": "fid1", "hash": _sha(b"img1")},
                {"file_id": "fid2", "hash": _sha(b"img2")},
            ],
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()) as up,
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess)
        up.assert_not_awaited()
        if sess.calls[0]["attachment_ids"] != ["fid1", "fid2"]:
            pytest.fail("follow-up missed prior images")
        # registry survives the turn unchanged
        if [e["file_id"] for e in sess.attachments] != ["fid1", "fid2"]:
            pytest.fail("registry changed unexpectedly")

    @staticmethod
    async def test_replayed_bytes_reuse_uploaded_id_without_reupload() -> None:
        """Verify replayed bytes reuse the uploaded id."""
        sess = FakeSess(attachments=[{"file_id": "fid1", "hash": _sha(b"img1")}])
        jobs: list[dict[str, bytes | str]] = [
            {"name": "img1.png", "data": b"img1", "mime": "image/png"},
        ]
        with (
            patch.object(server, "upload_file", new=AsyncMock()) as up,
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess, file_jobs=jobs)
        up.assert_not_awaited()
        if sess.calls[0]["attachment_ids"] != ["fid1"]:
            pytest.fail("replayed bytes missed registry id")

    @staticmethod
    async def test_new_bytes_are_uploaded_and_remembered() -> None:
        """Verify new bytes are uploaded and remembered."""
        sess = FakeSess()
        jobs: list[dict[str, bytes | str]] = [
            {"name": "new.png", "data": b"fresh", "mime": "image/png"},
        ]
        up = AsyncMock(return_value={"fileMetadataId": "fid-new"})
        with (
            patch.object(server, "upload_file", new=up),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess, file_jobs=jobs)
        if sess.calls[0]["attachment_ids"] != ["fid-new"]:
            pytest.fail("new upload id missing")
        if sess.attachments != [{"file_id": "fid-new", "hash": _sha(b"fresh")}]:
            pytest.fail("new upload not remembered")

    @staticmethod
    async def test_mixed_prior_and_new_all_mentioned() -> None:
        """Verify mixed prior and new uploads are all mentioned."""
        sess = FakeSess(attachments=[{"file_id": "fid1", "hash": _sha(b"img1")}])
        jobs: list[dict[str, bytes | str]] = [
            {"name": "img1.png", "data": b"img1", "mime": "image/png"},
            {"name": "img2.png", "data": b"img2", "mime": "image/png"},
        ]
        up = AsyncMock(return_value={"fileMetadataId": "fid2"})
        with (
            patch.object(server, "upload_file", new=up),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess, file_jobs=jobs)
        # img1 bytes hit the registry (no upload); only img2 uploaded once
        if up.await_count != 1:
            pytest.fail("unexpected upload count")
        if sess.calls[0]["attachment_ids"] != ["fid1", "fid2"]:
            pytest.fail("mixed ids missing")
        if [e["file_id"] for e in sess.attachments] != ["fid1", "fid2"]:
            pytest.fail("registry missed new id")

    @staticmethod
    async def test_mention_cap_keeps_most_recent() -> None:
        """Verify the mention cap keeps the most recent ids."""
        sess = FakeSess(
            attachments=[{"file_id": f"fid{i}", "hash": None} for i in range(1, 9)],
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()) as up,
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess)
        up.assert_not_awaited()
        if sess.calls[0]["attachment_ids"] != [
            "fid3",
            "fid4",
            "fid5",
            "fid6",
            "fid7",
            "fid8",
        ]:
            pytest.fail("mention cap kept wrong ids")


class StaleIdRetryTests(unittest.IsolatedAsyncioTestCase):
    """Cover stale file-id retry and drop propagation."""

    @staticmethod
    async def test_stale_ids_dropped_and_turn_retried() -> None:
        """Verify stale ids are dropped and the turn is retried."""
        sess = FakeSess(
            attachments=[{"file_id": "fid1", "hash": None}],
            script=[GatewayError("upstream", "FileAttachment not found: fid1"), None],
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            events = await _run_turn(sess)
        if not any(e["type"] == "done" for e in events):
            pytest.fail("retried turn missing done")
        if len(sess.calls) != _RETRY_CALL_COUNT:
            pytest.fail("turn was not retried once")
        if sess.calls[0]["attachment_ids"] != ["fid1"]:
            pytest.fail("first attempt missed id")
        if sess.calls[1]["attachment_ids"] is not None:
            pytest.fail("retry should send no ids")
        if sess.attachments != []:
            pytest.fail("stale id not dropped")

    @staticmethod
    async def test_passthrough_only_file_ids_are_mentioned() -> None:
        """file-id jobs without byte uploads must still be mentioned/remembered."""
        sess = FakeSess()
        jobs: list[dict[str, bytes | str]] = [{"file_id": "fid-x"}]
        with (
            patch.object(server, "upload_file", new=AsyncMock()) as up,
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess, file_jobs=jobs)
        up.assert_not_awaited()
        if sess.calls[0]["attachment_ids"] != ["fid-x"]:
            pytest.fail("passthrough id missing")
        if sess.attachments != [{"file_id": "fid-x", "hash": None}]:
            pytest.fail("passthrough id not remembered")

    @staticmethod
    async def test_turn_records_dropped_ids_for_propagation() -> None:
        """Verify the turn records dropped ids for propagation."""
        sess = FakeSess(
            attachments=[{"file_id": "fid1", "hash": None}],
            script=[GatewayError("upstream", "FileAttachment not found: fid1"), None],
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess)
        if sess.last_dropped_attachment_ids != {"fid1"}:
            pytest.fail("dropped ids not recorded")

    @staticmethod
    async def test_stale_retry_without_drop_leaves_marker_empty() -> None:
        """A midstream failure never retries -> no id may be declared stale."""
        sess = FakeSess(
            attachments=[{"file_id": "fid1", "hash": None}],
            script=["yield-then-raise"],
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
            pytest.raises(GatewayError),
        ):
            await _run_turn(sess)
        if sess.last_dropped_attachment_ids != set():
            pytest.fail("midstream failure marked drops")

    @staticmethod
    async def test_propagate_dropped_ids_to_source_checkpoint() -> None:
        """Verify dropped ids propagate to the source checkpoint."""
        source = GrokSession("ck", "uid")
        source.attachments = [
            {"file_id": "f1", "hash": None},
            {"file_id": "f2", "hash": None},
            {"file_id": "f3", "hash": None},
        ]
        fork = GrokSession("ck", "uid")
        fork.attachments = [
            {"file_id": "f2", "hash": None},
            {"file_id": "f3", "hash": None},
        ]
        fork.last_dropped_attachment_ids = {"f1"}
        server.propagate_dropped_attachments(source, fork)
        if [e["file_id"] for e in source.attachments] != ["f2", "f3"]:
            pytest.fail("dropped id not propagated")

    @staticmethod
    async def test_propagate_ignores_cap_evictions_and_additions() -> None:
        """Verify propagation ignores cap evictions and additions."""
        source = GrokSession("ck", "uid")
        source.attachments = [
            {"file_id": "f1", "hash": None},
            {"file_id": "f2", "hash": None},
        ]
        # fork lost f1 via registry cap and gained f-new: nothing declared
        # stale -> the checkpoint must stay untouched
        fork = GrokSession("ck", "uid")
        fork.attachments = [
            {"file_id": "f2", "hash": None},
            {"file_id": "f-new", "hash": None},
        ]
        fork.last_dropped_attachment_ids = set()
        server.propagate_dropped_attachments(source, fork)
        if [e["file_id"] for e in source.attachments] != ["f1", "f2"]:
            pytest.fail("checkpoint changed without drops")

    @staticmethod
    async def test_mention_cap_keeps_current_turn_dedupe_hits() -> None:
        """A hash-reuse hit for THIS turn's image must survive the 6-id cap."""
        registry: list[dict[str, str | None]] = [
            {"file_id": f"fid{i}", "hash": _sha(f"img{i}".encode())}
            for i in range(1, 9)
        ]
        sess = FakeSess(attachments=registry)
        # replay of turn-1 bytes (img1): dedupe hit on the oldest entry
        jobs: list[dict[str, bytes | str]] = [
            {"name": "img1.png", "data": b"img1", "mime": "image/png"},
        ]
        with (
            patch.object(server, "upload_file", new=AsyncMock()) as up,
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess, file_jobs=jobs)
        up.assert_not_awaited()
        sent = sess.calls[0]["attachment_ids"]
        if not isinstance(sent, list):
            pytest.fail("missing attachment ids")
        if len(sent) != server.MAX_TURN_MENTIONS:
            pytest.fail("mention cap wrong size")
        if "fid1" not in sent:  # current-turn image not evicted
            pytest.fail("dedupe hit evicted by cap")
        # LRU bump: the reused entry moved to most-recent in the registry
        if sess.attachments[-1]["file_id"] != "fid1":
            pytest.fail("reused entry not bumped")

    @staticmethod
    async def test_stale_retry_spares_capped_out_and_fresh_ids() -> None:
        """Only remembered ids actually mentioned this turn may be dropped."""
        sess = FakeSess(
            attachments=[{"file_id": f"fid{i}", "hash": None} for i in range(1, 9)],
            script=[GatewayError("upstream", "FileAttachment not found"), None],
        )
        jobs: list[dict[str, bytes | str]] = [
            {"name": "new.png", "data": b"fresh", "mime": "image/png"},
        ]
        with (
            patch.object(
                server,
                "upload_file",
                new=AsyncMock(return_value={"fileMetadataId": "fid-new"}),
            ),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess, file_jobs=jobs)
        kept = [e["file_id"] for e in sess.attachments]
        if "fid1" not in kept:  # never mentioned -> never dropped
            pytest.fail("unmentioned id dropped")
        if "fid2" not in kept:
            pytest.fail("unmentioned id dropped")
        if "fid-new" not in kept:  # fresh upload survives the stale sweep
            pytest.fail("fresh upload dropped")
        if "fid8" in kept:  # mentioned -> declared stale -> dropped
            pytest.fail("stale mentioned id kept")

    @staticmethod
    async def test_retry_keeps_fresh_uploads() -> None:
        """Verify a retry keeps fresh uploads."""
        sess = FakeSess(
            attachments=[{"file_id": "fid-old", "hash": None}],
            script=[GatewayError("upstream", "file attachment missing"), None],
        )
        jobs: list[dict[str, bytes | str]] = [
            {"name": "new.png", "data": b"fresh", "mime": "image/png"},
        ]
        with (
            patch.object(
                server,
                "upload_file",
                new=AsyncMock(return_value={"fileMetadataId": "fid-new"}),
            ),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess, file_jobs=jobs)
        if sess.calls[0]["attachment_ids"] != ["fid-old", "fid-new"]:
            pytest.fail("first attempt ids wrong")
        if sess.calls[1]["attachment_ids"] != ["fid-new"]:
            pytest.fail("retry ids wrong")
        if [e["file_id"] for e in sess.attachments] != ["fid-new"]:
            pytest.fail("registry wrong after retry")

    @staticmethod
    async def test_non_file_gateway_error_not_retried() -> None:
        """Verify a non-file gateway error is not retried."""
        sess = FakeSess(
            attachments=[{"file_id": "fid1", "hash": None}],
            script=[GatewayError("quota", "usage limit exceeded")],
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
            pytest.raises(GatewayError),
        ):
            await _run_turn(sess)
        if len(sess.calls) != 1:
            pytest.fail("non-file error retried")

    @staticmethod
    async def test_midstream_file_error_not_retried() -> None:
        """Verify a midstream file error is not retried."""
        sess = FakeSess(
            attachments=[{"file_id": "fid1", "hash": None}],
            script=["yield-then-raise"],
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
            pytest.raises(GatewayError),
        ):
            await _run_turn(sess)
        if len(sess.calls) != 1:
            pytest.fail("midstream error retried")

    @staticmethod
    async def test_failed_new_uploads_still_fail_the_turn() -> None:
        """Remembered ids must not mask a fully-failed fresh batch (662b0fc)."""
        sess = FakeSess(attachments=[{"file_id": "fid1", "hash": None}])
        jobs: list[dict[str, bytes | str]] = [
            {"name": "image.png", "data": b"x", "mime": "image/png"},
        ]
        with (
            patch.object(
                server,
                "upload_file",
                new=AsyncMock(side_effect=UploadError("init failed 403")),
            ),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
            pytest.raises(GatewayError) as exc_info,
        ):
            await _run_turn(sess, file_jobs=jobs)
        if "attachment upload failed" not in str(exc_info.value):
            pytest.fail("wrong failure message")
        if len(sess.calls) != 0:
            pytest.fail("failed upload reached gateway")


class RegistryCarryTests(unittest.TestCase):
    """Cover registry carry-over across checkpoints."""

    @staticmethod
    def test_clone_checkpoint_copies_registry() -> None:
        """Verify cloning a checkpoint copies the registry."""
        s1 = GrokSession("ck", "uid")
        s1.attachments = [{"file_id": "f1", "hash": "h1"}]
        s2 = s1.clone_checkpoint()
        if s2.attachments != [{"file_id": "f1", "hash": "h1"}]:
            pytest.fail("clone missed registry")
        s2.attachments.append({"file_id": "f2", "hash": None})
        if [e["file_id"] for e in s1.attachments] != ["f1"]:
            pytest.fail("clone shares registry")


class StoreRoundTripTests(unittest.TestCase):
    """Cover attachment registry persistence round-trips."""

    @override
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def test_attachments_roundtrip_and_preserve_on_none(self) -> None:
        """Verify attachments round-trip and survive a None save."""
        store = SqliteStore(Path(self.tmp_dir) / "t.db")
        self.addCleanup(store.close)
        reg = [
            {"file_id": "fid1", "hash": _sha(b"img1")},
            {"file_id": "fid2", "hash": None},
        ]
        store.save_session("k", "acc", ["u1"], attachments=reg)
        sess = store.get_session("k")
        if sess is None:
            pytest.fail("session missing")
        if sess["attachments"] != reg:
            pytest.fail("round-trip mismatch")
        # callsite that does not manage a registry must not wipe it
        store.save_session("k", "acc", ["u1"])
        sess = store.get_session("k")
        if sess is None:
            pytest.fail("session missing after resave")
        if sess["attachments"] != reg:
            pytest.fail("registry wiped on None save")
        # malformed entries are filtered
        store.save_session("k2", "acc", [], attachments=[{"nope": 1}])
        sess2 = store.get_session("k2")
        if sess2 is None:
            pytest.fail("malformed session missing")
        if sess2["attachments"] != []:
            pytest.fail("malformed entries not filtered")

    def test_legacy_db_without_column_is_migrated(self) -> None:
        """Verify a legacy database without the column is migrated."""
        path = Path(self.tmp_dir) / "legacy.db"
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE sessions (
                session_key TEXT PRIMARY KEY, account_key TEXT NOT NULL,
                user_chain_json TEXT NOT NULL DEFAULT '[]',
                conversation_id TEXT NOT NULL DEFAULT '',
                last_parent_response_id TEXT NOT NULL DEFAULT '',
                model_mode TEXT NOT NULL DEFAULT 'fast',
                created_at REAL NOT NULL, last_used REAL NOT NULL);
            INSERT INTO sessions VALUES ('old', 'a', '[]', 'c', 'p', 'fast', 1.0, 2.0);
        """)
        conn.commit()
        conn.close()
        store = SqliteStore(path)
        self.addCleanup(store.close)
        old = store.get_session("old")
        if old is None:
            pytest.fail("legacy session missing")
        if old["attachments"] != []:
            pytest.fail("legacy session should have no attachments")
        store.save_session(
            "new",
            "a",
            [],
            attachments=[{"file_id": "f2", "hash": None}],
        )
        new = store.get_session("new")
        if new is None:
            pytest.fail("new session missing")
        if new["attachments"] != [{"file_id": "f2", "hash": None}]:
            pytest.fail("new session attachments wrong")


if __name__ == "__main__":
    unittest.main()
