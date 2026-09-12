# Copyright (c) 2026 grok-to-openai-api contributors.
"""Regression tests for Responses API input_text content parts.

llmcord-go sends image-attachment messages on /v1/responses with content
parts [{type: input_text}, {type: input_image}]. extract_attachments and
content_to_text only recognized the Chat-Completions text part type, so the
prompt arrived empty while the attachment survived -- grok then answered
bare-image style instead of editing.

Run: python3 -m unittest -v tests.test_responses_input_text
"""

from __future__ import annotations

import base64
import json
import unittest
from types import SimpleNamespace
from typing import TypedDict, override
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request

import server
from grok_gateway import TurnResult

TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


class _LlmcordBody(TypedDict):
    """Typed body for the llmcord vision fixture."""

    model: str
    input: list[dict[str, object]]


def llmcord_body() -> _LlmcordBody:
    """Return the exact llmcord-go vision payload.

    Role-only input items without a type key whose content lists
    input_text and input_image parts.

    Returns:
        Request body dict for the Responses endpoint.

    """
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
            },
        ],
    }


class FakeRequest(Request):
    """Real Request carrying a canned JSON body."""

    def __init__(self, body: object) -> None:
        """Initialize the fake with a JSON-serializable body.

        Args:
            body: Request payload returned by ``json`` and ``body``.

        """
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
    async def json(self) -> object:
        return self._body_data

    @override
    async def body(self) -> bytes:
        return json.dumps(self._body_data).encode()


class ContentToTextTests(unittest.TestCase):
    """Verify content_to_text flattens Responses input_text parts."""

    @staticmethod
    def test_input_text_parts_flatten() -> None:
        """Verify multiple input_text parts join with newlines."""
        parts: list[dict[str, object]] = [
            {"type": "input_text", "text": "hello"},
            {"type": "input_text", "text": "world"},
        ]
        if server.content_to_text(parts) != "hello\nworld":
            pytest.fail("input_text parts not flattened")

    @staticmethod
    def test_chat_completions_text_parts_still_flatten() -> None:
        """Verify legacy text parts still flatten."""
        if server.content_to_text([{"type": "text", "text": "hello"}]) != "hello":
            pytest.fail("legacy text part not flattened")

    @staticmethod
    def test_mixed_input_text_and_image_parts_flatten_text_only() -> None:
        """Verify image parts are skipped while text is kept."""
        parts: list[dict[str, object]] = [
            {"type": "input_text", "text": "edit this"},
            {"type": "input_image", "image_url": "https://x/y.png"},
        ]
        if server.content_to_text(parts) != "edit this":
            pytest.fail("mixed parts not flattened")


class ExtractAttachmentsTests(unittest.IsolatedAsyncioTestCase):
    """Verify extract_attachments keeps input_text alongside files."""

    @staticmethod
    async def test_input_text_survives_extraction_alongside_image() -> None:
        """Verify input_text survives extraction alongside an image."""
        msgs = llmcord_body()["input"]
        out, jobs = await server.extract_attachments(msgs)
        if out[0]["content"] != "give this cat a cowboy hat.":
            pytest.fail("input_text lost during extraction")
        if len(jobs) != 1:
            pytest.fail("expected one file job")
        if "data" not in jobs[0]:
            pytest.fail("file job missing data")

    @staticmethod
    async def test_chat_completions_text_part_still_extracts() -> None:
        """Verify legacy text and image_url parts still extract."""
        msgs: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + TINY_PNG_B64},
                    },
                ],
            },
        ]
        out, jobs = await server.extract_attachments(msgs)
        if out[0]["content"] != "hi":
            pytest.fail("legacy text lost during extraction")
        if len(jobs) != 1:
            pytest.fail("expected one file job")

    @staticmethod
    async def test_llmcord_input_file_part_uploads() -> None:
        """Verify top-level input_file parts become upload jobs."""
        # responses.go sends {type: input_file, file_data, filename} at the
        # part top level; the chat-nested file object is absent.
        raw = b"hello doc"
        data_url = "data:text/plain;base64," + base64.b64encode(raw).decode()
        msgs: list[dict[str, object]] = [
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
            },
        ]
        out, jobs = await server.extract_attachments(msgs)
        if out[0]["content"] != "summarize":
            pytest.fail("input_text lost with input_file")
        if len(jobs) != 1:
            pytest.fail("expected one file job")
        if jobs[0]["name"] != "notes.txt":
            pytest.fail("file job name mismatch")
        if jobs[0]["data"] != raw:
            pytest.fail("file job data mismatch")
        if jobs[0]["mime"] != "text/plain":
            pytest.fail("file job mime mismatch")

    @staticmethod
    async def test_input_file_part_without_usable_fields_is_dropped() -> None:
        """Verify an empty input_file part yields no jobs."""
        msgs: list[dict[str, object]] = [
            {"role": "user", "content": [{"type": "input_file"}]},
        ]
        _out, jobs = await server.extract_attachments(msgs)
        if jobs != []:
            pytest.fail("expected no file jobs")


def _check_response_text_ok(body: bytes) -> None:
    """Verify the Responses body carries the expected ok text.

    Args:
        body: Raw response body bytes.

    """
    data: object = json.loads(body)
    if not isinstance(data, dict):
        pytest.fail("response body is not a dict")
    output = data.get("output")
    if not isinstance(output, list) or not output:
        pytest.fail("response output missing")
    first = output[0]
    if not isinstance(first, dict):
        pytest.fail("response output item malformed")
    content = first.get("content")
    if not isinstance(content, list) or not content:
        pytest.fail("response content missing")
    item = content[0]
    if not isinstance(item, dict):
        pytest.fail("response content item malformed")
    if item.get("text") != "ok":
        pytest.fail("response text mismatch")


class ResponsesApiInputTextEndpointTest(unittest.IsolatedAsyncioTestCase):
    """Verify the Responses endpoint forwards input_text prompts."""

    @staticmethod
    async def test_prompt_includes_input_text_alongside_attachment() -> None:
        """Verify the endpoint prompt includes input_text with files."""
        fake_acc = SimpleNamespace(key="u:k", cookie_header=lambda: "ck")
        fake_state = SimpleNamespace(
            grok=SimpleNamespace(
                conversation_id="conv-1",
                last_parent_response_id="parent-1",
                attachments=[],
                last_dropped_attachment_ids=set(),
            ),
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
        if await_args is None:
            pytest.fail("pick_account_and_turn not awaited")
        if await_args.args[1] != ["give this cat a cowboy hat."]:
            pytest.fail("users list mismatch")
        kwargs = await_args.kwargs
        if kwargs["prompt"] != "give this cat a cowboy hat.":
            pytest.fail("prompt mismatch")
        if len(kwargs["file_jobs"]) != 1:
            pytest.fail("expected one file job")
        _check_response_text_ok(bytes(resp.body))


class ResponsesApiShorthandItemsTest(unittest.IsolatedAsyncioTestCase):
    """Verify shorthand input items fold into one user message.

    Top-level shorthand sequences must fold into one user message because
    prompt extraction takes the last user message, so separate messages
    would lose the text.
    """

    @staticmethod
    async def _run(input_items: list[dict[str, object]]) -> AsyncMock:
        """Run the Responses endpoint with shorthand input items.

        Args:
            input_items: Raw input list sent as the request input.

        Returns:
            Mocked pick_account_and_turn capturing the turn arguments.

        """
        fake_acc = SimpleNamespace(key="u:k", cookie_header=lambda: "ck")
        fake_state = SimpleNamespace(
            grok=SimpleNamespace(
                conversation_id="conv-1",
                last_parent_response_id="parent-1",
                attachments=[],
                last_dropped_attachment_ids=set(),
            ),
        )
        fake_result = TurnResult(
            text="ok",
            response_id="r1",
            conversation_id="conv-1",
            parent_response_id="parent-1",
        )
        mock_turn = AsyncMock(return_value=(fake_acc, fake_result, [], fake_state))
        body: dict[str, object] = {"model": "grok-fast", "input": input_items}
        with (
            patch("server.pick_account_and_turn", new=mock_turn),
            patch("server.refresh_statsig_pair", new=AsyncMock()),
            patch("server.host_images", new=AsyncMock(return_value=[])),
        ):
            await server.responses_api(FakeRequest(body))
        return mock_turn

    async def test_bare_input_text_then_image_items_keep_prompt(self) -> None:
        """Verify bare text plus image items keep the prompt."""
        mock_turn = await self._run(
            [
                {"type": "input_text", "text": "give this cat a cowboy hat."},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + TINY_PNG_B64,
                },
            ],
        )
        await_args = mock_turn.await_args
        if await_args is None:
            pytest.fail("pick_account_and_turn not awaited")
        kwargs = await_args.kwargs
        if kwargs["prompt"] != "give this cat a cowboy hat.":
            pytest.fail("prompt mismatch")
        if len(kwargs["file_jobs"]) != 1:
            pytest.fail("expected one file job")

    async def test_role_message_then_shorthand_image_keeps_prompt(self) -> None:
        """Verify a role message plus shorthand image keeps the prompt."""
        mock_turn = await self._run(
            [
                {"role": "user", "content": "look at this"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + TINY_PNG_B64,
                },
            ],
        )
        await_args = mock_turn.await_args
        if await_args is None:
            pytest.fail("pick_account_and_turn not awaited")
        kwargs = await_args.kwargs
        if kwargs["prompt"] != "look at this":
            pytest.fail("prompt mismatch")
        if len(kwargs["file_jobs"]) != 1:
            pytest.fail("expected one file job")


if __name__ == "__main__":
    unittest.main()
