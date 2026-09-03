"""Regression tests: Responses API `input_text` content parts must reach the prompt.

llmcord-go sends image-attachment messages on /v1/responses with content parts
[{type: "input_text", ...}, {type: "input_image", ...}]. extract_attachments and
content_to_text only recognized the Chat-Completions "text" part type, so the
prompt arrived empty while the attachment survived — grok then answered
bare-image style ("adorable cat, anything specific?") instead of editing
(2026-08-31 10:46 PHT "give this cat a cowboy hat." incident; the persisted
session had user_chain_json=[""]).
Run: python3 -m unittest -v tests.test_responses_input_text
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any, override
from unittest.mock import AsyncMock, patch

from fastapi import Request

import server
from grok_gateway import TurnResult

TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def llmcord_body() -> dict[str, Any]:
    """Exact shape llmcord-go sends for a vision message on /v1/responses:
    role-only input items (no "type" key) whose content is a list of
    input_text / input_image parts (responses.go:402-405, 553-577)."""
    return {
        "model": "grok-fast",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "give this cat a cowboy hat."},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64," + TINY_PNG_B64,
                    },
                ],
            }
        ],
    }


class FakeRequest(Request):
    """Real Request carrying a canned JSON body."""

    def __init__(self, body: dict[str, Any]) -> None:
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [],
            "query_string": b"",
            "server": ("test", 80),
            "scheme": "http",
            "client": ("test", 50000),
        }
        super().__init__(scope)
        self._body_data = body

    @override
    async def json(self) -> Any:
        return self._body_data

    @override
    async def body(self) -> bytes:
        return json.dumps(self._body_data).encode()


class ContentToTextTests(unittest.TestCase):
    def test_input_text_parts_flatten(self):
        parts = [
            {"type": "input_text", "text": "hello"},
            {"type": "input_text", "text": "world"},
        ]
        self.assertEqual(server.content_to_text(parts), "hello\nworld")

    def test_chat_completions_text_parts_still_flatten(self):
        self.assertEqual(
            server.content_to_text([{"type": "text", "text": "hello"}]), "hello"
        )

    def test_mixed_input_text_and_image_parts_flatten_text_only(self):
        parts = [
            {"type": "input_text", "text": "edit this"},
            {"type": "input_image", "image_url": "https://x/y.png"},
        ]
        self.assertEqual(server.content_to_text(parts), "edit this")


class ExtractAttachmentsTests(unittest.IsolatedAsyncioTestCase):
    async def test_input_text_survives_extraction_alongside_image(self):
        msgs = llmcord_body()["input"]
        out, jobs = await server.extract_attachments(msgs)
        self.assertEqual(out[0]["content"], "give this cat a cowboy hat.")
        self.assertEqual(len(jobs), 1)
        self.assertIn("data", jobs[0])

    async def test_chat_completions_text_part_still_extracts(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + TINY_PNG_B64},
                    },
                ],
            }
        ]
        out, jobs = await server.extract_attachments(msgs)
        self.assertEqual(out[0]["content"], "hi")
        self.assertEqual(len(jobs), 1)

    async def test_llmcord_input_file_part_uploads(self):
        # responses.go:588-601 sends {type: "input_file",
        # file_data: "data:<mime>;base64,...", filename} at the part top
        # level; the chat-nested "file" object is absent.
        import base64

        raw = b"hello doc"
        data_url = "data:text/plain;base64," + base64.b64encode(raw).decode()
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "summarize"},
                    {
                        "type": "input_file",
                        "file_data": data_url,
                        "filename": "notes.txt",
                    },
                ],
            }
        ]
        out, jobs = await server.extract_attachments(msgs)
        self.assertEqual(out[0]["content"], "summarize")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["name"], "notes.txt")
        self.assertEqual(jobs[0]["data"], raw)
        self.assertEqual(jobs[0]["mime"], "text/plain")

    async def test_input_file_part_without_usable_fields_is_dropped_loudly(self):
        msgs = [{"role": "user", "content": [{"type": "input_file"}]}]
        out, jobs = await server.extract_attachments(msgs)
        self.assertEqual(jobs, [])


class ResponsesApiInputTextEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_includes_input_text_alongside_attachment(self):
        fake_acc = SimpleNamespace(key="u:k", cookie_header=lambda: "ck")
        fake_state = SimpleNamespace(
            grok=SimpleNamespace(
                conversation_id="conv-1",
                last_parent_response_id="parent-1",
                attachments=[],
                last_dropped_attachment_ids=set(),
            )
        )
        fake_result = TurnResult(
            text="ok",
            response_id="r1",
            conversation_id="conv-1",
            parent_response_id="parent-1",
        )
        mock_turn = AsyncMock(return_value=(fake_acc, fake_result, [], fake_state))
        with (
            patch("server.pick_account_and_turn", new=mock_turn),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
            patch("server.host_images", new=AsyncMock(return_value=[])),
        ):
            resp = await server.responses_api(FakeRequest(llmcord_body()))

        await_args = mock_turn.await_args
        assert await_args is not None
        self.assertEqual(await_args.args[1], ["give this cat a cowboy hat."])
        kwargs = await_args.kwargs
        self.assertEqual(kwargs["prompt"], "give this cat a cowboy hat.")
        self.assertEqual(len(kwargs["file_jobs"]), 1)
        data = json.loads(resp.body)
        self.assertEqual(data["output"][0]["content"][0]["text"], "ok")


class ResponsesApiShorthandItemsTest(unittest.IsolatedAsyncioTestCase):
    """Top-level shorthand sequences ({input_text}, {input_image}) must fold
    into one user message: prompt extraction takes the last user message, so
    separate messages would lose the text (review residual, 2026-08-31)."""

    async def _run(self, input_items):
        fake_acc = SimpleNamespace(key="u:k", cookie_header=lambda: "ck")
        fake_state = SimpleNamespace(
            grok=SimpleNamespace(
                conversation_id="conv-1",
                last_parent_response_id="parent-1",
                attachments=[],
                last_dropped_attachment_ids=set(),
            )
        )
        fake_result = TurnResult(
            text="ok",
            response_id="r1",
            conversation_id="conv-1",
            parent_response_id="parent-1",
        )
        mock_turn = AsyncMock(return_value=(fake_acc, fake_result, [], fake_state))
        body = {"model": "grok-fast", "input": input_items}
        with (
            patch("server.pick_account_and_turn", new=mock_turn),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
            patch("server.host_images", new=AsyncMock(return_value=[])),
        ):
            await server.responses_api(FakeRequest(body))
        return mock_turn

    async def test_bare_input_text_then_image_items_keep_prompt(self):
        mock_turn = await self._run(
            [
                {"type": "input_text", "text": "give this cat a cowboy hat."},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + TINY_PNG_B64,
                },
            ]
        )
        await_args = mock_turn.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        self.assertEqual(kwargs["prompt"], "give this cat a cowboy hat.")
        self.assertEqual(len(kwargs["file_jobs"]), 1)

    async def test_role_message_then_shorthand_image_keeps_prompt(self):
        mock_turn = await self._run(
            [
                {"role": "user", "content": "look at this"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + TINY_PNG_B64,
                },
            ]
        )
        await_args = mock_turn.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        self.assertEqual(kwargs["prompt"], "look at this")
        self.assertEqual(len(kwargs["file_jobs"]), 1)


if __name__ == "__main__":
    unittest.main()
