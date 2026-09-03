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
import os
import shutil
import sqlite3
import tempfile
import unittest
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, override
from unittest.mock import AsyncMock, patch

import server
from grok_gateway import GatewayError, GrokSession
from session_store import SqliteStore
from uploads import UploadError


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeSess(GrokSession):
    """Records ask() calls; scripted behaviors per call.

    Script items: None -> normal done; an Exception -> raise it;
    "yield-then-raise" -> emit one delta, then raise a file error.
    """

    def __init__(
        self,
        attachments: list[dict[str, Any]] | None = None,
        script: list[Any] | None = None,
    ) -> None:
        super().__init__("", "", "fast")
        self.cookie_header = "ck"
        self.attachments = list(attachments or [])
        self.calls: list[dict[str, Any]] = []
        self._script: list[Any] = list(script or [])

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
    ) -> AsyncIterator[dict[str, Any]]:
        self.calls.append({"prompt": prompt, "attachment_ids": attachment_ids})
        behavior = self._script.pop(0) if self._script else None
        if behavior == "yield-then-raise":
            yield {"type": "text_delta", "text": "partial"}
            raise GatewayError("upstream", "FileAttachment not found: fid1")
        if isinstance(behavior, Exception):
            raise behavior
        yield {"type": "done", "result": SimpleNamespace(text="ok")}


async def _run_turn(sess, prompt="top 3", **kwargs):
    return [ev async for ev in server.stream_session_turn(sess, prompt, **kwargs)]


class ReMentionTests(unittest.IsolatedAsyncioTestCase):
    async def test_followup_turn_rementions_prior_images(self):
        """The incident: a follow-up with no attachments re-mentions turn-1 files."""
        sess = FakeSess(
            attachments=[
                {"file_id": "fid1", "hash": _sha(b"img1")},
                {"file_id": "fid2", "hash": _sha(b"img2")},
            ]
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()) as up,
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess)
        up.assert_not_awaited()
        self.assertEqual(sess.calls[0]["attachment_ids"], ["fid1", "fid2"])
        # registry survives the turn unchanged
        self.assertEqual([e["file_id"] for e in sess.attachments], ["fid1", "fid2"])

    async def test_replayed_bytes_reuse_uploaded_id_without_reupload(self):
        sess = FakeSess(attachments=[{"file_id": "fid1", "hash": _sha(b"img1")}])
        jobs = [{"name": "img1.png", "data": b"img1", "mime": "image/png"}]
        with (
            patch.object(server, "upload_file", new=AsyncMock()) as up,
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess, file_jobs=jobs)
        up.assert_not_awaited()
        self.assertEqual(sess.calls[0]["attachment_ids"], ["fid1"])

    async def test_new_bytes_are_uploaded_and_remembered(self):
        sess = FakeSess()
        jobs = [{"name": "new.png", "data": b"fresh", "mime": "image/png"}]
        up = AsyncMock(return_value={"fileMetadataId": "fid-new"})
        with (
            patch.object(server, "upload_file", new=up),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess, file_jobs=jobs)
        self.assertEqual(sess.calls[0]["attachment_ids"], ["fid-new"])
        self.assertEqual(
            sess.attachments, [{"file_id": "fid-new", "hash": _sha(b"fresh")}]
        )

    async def test_mixed_prior_and_new_all_mentioned(self):
        sess = FakeSess(attachments=[{"file_id": "fid1", "hash": _sha(b"img1")}])
        jobs = [
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
        self.assertEqual(up.await_count, 1)
        self.assertEqual(sess.calls[0]["attachment_ids"], ["fid1", "fid2"])
        self.assertEqual([e["file_id"] for e in sess.attachments], ["fid1", "fid2"])

    async def test_mention_cap_keeps_most_recent(self):
        sess = FakeSess(
            attachments=[{"file_id": f"fid{i}", "hash": None} for i in range(1, 9)]
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()) as up,
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess)
        up.assert_not_awaited()
        self.assertEqual(
            sess.calls[0]["attachment_ids"],
            ["fid3", "fid4", "fid5", "fid6", "fid7", "fid8"],
        )


class StaleIdRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_ids_dropped_and_turn_retried(self):
        sess = FakeSess(
            attachments=[{"file_id": "fid1", "hash": None}],
            script=[GatewayError("upstream", "FileAttachment not found: fid1"), None],
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            events = await _run_turn(sess)
        self.assertTrue(any(e["type"] == "done" for e in events))
        self.assertEqual(len(sess.calls), 2)
        self.assertEqual(sess.calls[0]["attachment_ids"], ["fid1"])
        self.assertIsNone(sess.calls[1]["attachment_ids"])
        self.assertEqual(sess.attachments, [])

    async def test_passthrough_only_file_ids_are_mentioned(self):
        """file-id jobs without byte uploads must still be mentioned/remembered."""
        sess = FakeSess()
        jobs = [{"file_id": "fid-x"}]
        with (
            patch.object(server, "upload_file", new=AsyncMock()) as up,
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess, file_jobs=jobs)
        up.assert_not_awaited()
        self.assertEqual(sess.calls[0]["attachment_ids"], ["fid-x"])
        self.assertEqual(sess.attachments, [{"file_id": "fid-x", "hash": None}])

    async def test_turn_records_dropped_ids_for_propagation(self):
        sess = FakeSess(
            attachments=[{"file_id": "fid1", "hash": None}],
            script=[GatewayError("upstream", "FileAttachment not found: fid1"), None],
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess)
        self.assertEqual(sess.last_dropped_attachment_ids, {"fid1"})

    async def test_stale_retry_without_drop_leaves_marker_empty(self):
        """A midstream failure never retries -> no id may be declared stale."""
        sess = FakeSess(
            attachments=[{"file_id": "fid1", "hash": None}], script=["yield-then-raise"]
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            with self.assertRaises(GatewayError):
                await _run_turn(sess)
        self.assertEqual(sess.last_dropped_attachment_ids, set())

    async def test_propagate_dropped_ids_to_source_checkpoint(self):
        from types import SimpleNamespace as NS

        source = NS(
            attachments=[
                {"file_id": "f1", "hash": None},
                {"file_id": "f2", "hash": None},
                {"file_id": "f3", "hash": None},
            ]
        )
        fork = NS(
            attachments=[
                {"file_id": "f2", "hash": None},
                {"file_id": "f3", "hash": None},
            ],
            last_dropped_attachment_ids={"f1"},
        )
        server._propagate_dropped_attachments(source, fork)
        self.assertEqual([e["file_id"] for e in source.attachments], ["f2", "f3"])

    async def test_propagate_ignores_cap_evictions_and_additions(self):
        from types import SimpleNamespace as NS

        source = NS(
            attachments=[
                {"file_id": "f1", "hash": None},
                {"file_id": "f2", "hash": None},
            ]
        )
        # fork lost f1 via registry cap and gained f-new: nothing declared
        # stale -> the checkpoint must stay untouched
        fork = NS(
            attachments=[
                {"file_id": "f2", "hash": None},
                {"file_id": "f-new", "hash": None},
            ],
            last_dropped_attachment_ids=set(),
        )
        server._propagate_dropped_attachments(source, fork)
        self.assertEqual([e["file_id"] for e in source.attachments], ["f1", "f2"])

    async def test_mention_cap_keeps_current_turn_dedupe_hits(self):
        """A hash-reuse hit for THIS turn's image must survive the 6-id cap."""
        registry = [
            {"file_id": f"fid{i}", "hash": _sha(f"img{i}".encode())}
            for i in range(1, 9)
        ]
        sess = FakeSess(attachments=registry)
        # replay of turn-1 bytes (img1): dedupe hit on the oldest entry
        jobs = [{"name": "img1.png", "data": b"img1", "mime": "image/png"}]
        with (
            patch.object(server, "upload_file", new=AsyncMock()) as up,
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess, file_jobs=jobs)
        up.assert_not_awaited()
        sent = sess.calls[0]["attachment_ids"]
        self.assertEqual(len(sent), server.MAX_TURN_MENTIONS)
        self.assertIn("fid1", sent)  # current-turn image not evicted
        # LRU bump: the reused entry moved to most-recent in the registry
        self.assertEqual(sess.attachments[-1]["file_id"], "fid1")

    async def test_stale_retry_spares_capped_out_and_fresh_ids(self):
        """Only remembered ids actually mentioned this turn may be dropped."""
        sess = FakeSess(
            attachments=[{"file_id": f"fid{i}", "hash": None} for i in range(1, 9)],
            script=[GatewayError("upstream", "FileAttachment not found"), None],
        )
        jobs = [{"name": "new.png", "data": b"fresh", "mime": "image/png"}]
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
        self.assertIn("fid1", kept)  # never mentioned -> never dropped
        self.assertIn("fid2", kept)
        self.assertIn("fid-new", kept)  # fresh upload survives the stale sweep
        self.assertNotIn("fid8", kept)  # mentioned -> declared stale -> dropped

    async def test_retry_keeps_fresh_uploads(self):
        sess = FakeSess(
            attachments=[{"file_id": "fid-old", "hash": None}],
            script=[GatewayError("upstream", "file attachment missing"), None],
        )
        jobs = [{"name": "new.png", "data": b"fresh", "mime": "image/png"}]
        with (
            patch.object(
                server,
                "upload_file",
                new=AsyncMock(return_value={"fileMetadataId": "fid-new"}),
            ),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            await _run_turn(sess, file_jobs=jobs)
        self.assertEqual(sess.calls[0]["attachment_ids"], ["fid-old", "fid-new"])
        self.assertEqual(sess.calls[1]["attachment_ids"], ["fid-new"])
        self.assertEqual([e["file_id"] for e in sess.attachments], ["fid-new"])

    async def test_non_file_gateway_error_not_retried(self):
        sess = FakeSess(
            attachments=[{"file_id": "fid1", "hash": None}],
            script=[GatewayError("quota", "usage limit exceeded")],
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            with self.assertRaises(GatewayError):
                await _run_turn(sess)
        self.assertEqual(len(sess.calls), 1)

    async def test_midstream_file_error_not_retried(self):
        sess = FakeSess(
            attachments=[{"file_id": "fid1", "hash": None}], script=["yield-then-raise"]
        )
        with (
            patch.object(server, "upload_file", new=AsyncMock()),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            with self.assertRaises(GatewayError):
                await _run_turn(sess)
        self.assertEqual(len(sess.calls), 1)

    async def test_failed_new_uploads_still_fail_the_turn(self):
        """Remembered ids must not mask a fully-failed fresh batch (662b0fc)."""
        sess = FakeSess(attachments=[{"file_id": "fid1", "hash": None}])
        jobs = [{"name": "image.png", "data": b"x", "mime": "image/png"}]
        with (
            patch.object(
                server,
                "upload_file",
                new=AsyncMock(side_effect=UploadError("init failed 403")),
            ),
            patch.object(server, "refresh_statsig_pair", new=AsyncMock()),
        ):
            with self.assertRaises(GatewayError) as ctx:
                await _run_turn(sess, file_jobs=jobs)
        self.assertIn("attachment upload failed", str(ctx.exception))
        self.assertEqual(len(sess.calls), 0)


class RegistryCarryTests(unittest.TestCase):
    def test_clone_checkpoint_copies_registry(self):
        s1 = GrokSession("ck", "uid")
        s1.attachments = [{"file_id": "f1", "hash": "h1"}]
        s2 = s1.clone_checkpoint()
        self.assertEqual(s2.attachments, [{"file_id": "f1", "hash": "h1"}])
        s2.attachments.append({"file_id": "f2", "hash": None})
        self.assertEqual([e["file_id"] for e in s1.attachments], ["f1"])


class StoreRoundTripTests(unittest.TestCase):
    @override
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def test_attachments_roundtrip_and_preserve_on_none(self):
        store = SqliteStore(os.path.join(self.tmp_dir, "t.db"))
        self.addCleanup(store.close)
        reg = [
            {"file_id": "fid1", "hash": _sha(b"img1")},
            {"file_id": "fid2", "hash": None},
        ]
        store.save_session("k", "acc", ["u1"], attachments=reg)
        sess = store.get_session("k")
        assert sess is not None
        self.assertEqual(sess["attachments"], reg)
        # callsite that does not manage a registry must not wipe it
        store.save_session("k", "acc", ["u1"])
        sess = store.get_session("k")
        assert sess is not None
        self.assertEqual(sess["attachments"], reg)
        # malformed entries are filtered
        raw: list[Any] = [{"nope": 1}, "junk", None]
        attachments: list[dict[str, Any]] = [e for e in raw if isinstance(e, dict)]
        store.save_session("k2", "acc", [], attachments=attachments)
        sess2 = store.get_session("k2")
        assert sess2 is not None
        self.assertEqual(sess2["attachments"], [])

    def test_legacy_db_without_column_is_migrated(self):
        path = os.path.join(self.tmp_dir, "legacy.db")
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
        assert old is not None
        self.assertEqual(old["attachments"], [])
        store.save_session(
            "new", "a", [], attachments=[{"file_id": "f2", "hash": None}]
        )
        new = store.get_session("new")
        assert new is not None
        self.assertEqual(new["attachments"], [{"file_id": "f2", "hash": None}])


if __name__ == "__main__":
    unittest.main()
