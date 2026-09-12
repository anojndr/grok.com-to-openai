# Repository Guidelines

## Project Overview

Turn `grok.com` free-account cookies (`accounts.txt`) into an OpenAI-compatible API via FastAPI. Supports `POST /v1/chat/completions` + `POST /v1/responses` (stream/non-stream, attachments, image-gen, multi-turn, `previous_response_id` chaining).

Entry: `python3 server.py` → `uvicorn.run(app, host=config.HOST, port=config.PORT)` (default `45080`).

## Architecture & Data Flow

Flat-module service, no `src/` layout. `server.py` orchestrates; `grok_gateway.py` owns Grok protocol; state in module singletons (`SESSIONS` dict + `SqliteStore` + `AccountPool`), no DI framework.

Request lifecycle (Grok → OpenAI):

1. `check_auth` (Bearer vs `G2O_API_KEY`) → `_read_json_body` → `_build_chat_context()` / `_build_responses_context()`: validate messages/`input`, `extract_attachments` (splits `image_url`/`file`/`input_image`/`input_file`/data-URL/remote-URL into `flat_messages` + `file_jobs`; text inlined via `inline_textual`), `resolve_mode(model)` (`MODEL_MODE_MAP`, default `grok-fast`), `chain_key(users, auth)` + `rid_hint`.
2. Session routing: `SESSIONS: dict[str, SessionState]` keyed by chain prefix. Hit → `fork_session_state` (live WS moves to fork, only newest message sent; gateway holds history). Miss → `_stream_fresh_turn` resends full transcript (`build_history_prompt`). SQLite mirrors checkpoints for restart recovery.
3. Turn: `stream_session_turn` / `run_session_turn` → `GrokSession.ask` / `_stream_turn`: `_ensure_connected`, `_upload_turn_files` (`asyncio.gather`, dedupe by sha, cap 6/turn, 12 stored), `_send_turn_openers` (`session.create` + `conversation.item.create`), frame loop (`_receive_turn_frame`) → `_apply_turn_event` → `TurnResult`.
4. Failover: `pick_account_and_turn` / `pick_account_and_stream_turn` loop `pool.acquire(exclude=tried)`; on `GatewayError` → `pool.release_fail(acc, _fail_kind(e))` + map via `_gateway_http_status` to `HTTPException`. Imagine path (`grok_generate_image`) rotates accounts independently.
5. Render: chat SSE (`_chat_chunk` + `data: [DONE]`) / `chat.completion` JSON; responses SSE via `_ResponseEventWriter.evt`. Images hosted (`host_images` → freeimage.host) as Markdown; optional `source_appendix` when `include_sources` or `G2O_INCLUDE_SOURCES=1`.

Key modules:

- `server.py` (~4816 lines): `app`, all routes (`chat_completions`, `responses_api`, `models`, `healthz`), context dataclasses (`_ChatContext`, `_ResponsesContext`, `_TurnRequest`), SSE renderers (`_chat_sse_frames`, `_responses_sse_frames`).
- `grok_gateway.py`: `GrokSession` (`connect`/`close`/`alive`/`ask`/`clone_checkpoint`, per-session `lock`), `GatewayError(kind)` (`upstream|timeout|closed|degraded|auth|quota`), `TurnResult`, `RenderFilter/strip_render_tags`, `grok_generate_image`.
- `accounts.py`: `Account` / `AccountPool` (`reload_if_changed`, `acquire`, `acquire_by_key`, `release_ok`, `release_fail`, 1h `degraded` quarantine).
- `session_store.py`: `SqliteStore` (WAL mode, sessions/checkpoints, account states, uid + statsig caches), `MAX_TRACKED_ATTACHMENTS=12`.
- `statsig.py`: `StatsigGenerator` (`ready`/`ensure_pair`/`generate`) for `x-statsig-id` anti-bot header.
- `uploads.py`: `upload_file` (presigned v2), `decode_data_url`/`guess_mime`, `freeimage_upload`, SSRF guards; all-failed upload = hard error.
- `config.py`: tiny `.env` loader + `G2O_*` defaults.

## Key Directories

- `/` (repo root): all runtime modules live here — no `src/`, `scripts/`, or `docs/`.
- `tests/`: stdlib `unittest` suite (9 files, see Testing & QA).
- `tools/`: one-off debug helpers only, e.g. `tools/capture_edit.py` (capture raw mgw frames to JSONL).
- `data/`: gitignored SQLite runtime state (`grok_store.db*`, WAL/SHM). Do not commit.
- Root artifacts (gitignored, runtime): `.env`, `accounts.txt`, `server.log`, `.server.pid`.

## Development Commands

No build step, no Docker/CI. Direct interpreter execution.

```bash
pip install -r requirements.txt          # or: uv sync (uv.lock is canonical)
cp .env.example .env                    # set FREEIMAGE_API_KEY, optionally G2O_API_KEY
python3 server.py                       # serves on $G2O_PORT (default 45080)
./restart.sh                            # prod-style: kills .server.pid/port owner, uvicorn server:app, waits on /healthz, tails server.log
python3 -m uvicorn server:app --host 0.0.0.0 --port 45080
curl -sf localhost:45080/healthz        # readiness: accounts/available/statsig_ready/sessions/degraded
```

Lint/typecheck (strict, `ruff select=["ALL"]`, `ty all="error"`):

Always use https://docs.astral.sh/ruff/ with everything enabled and https://docs.astral.sh/ty/ with everything enabled, then fix all of the issues. Make sure to actually fix all of the issues instead of suppressing them.

```bash
ruff check . && ruff format --check .
ty check
```

## Code Conventions & Common Patterns

- Formatting: `ruff`, `target-version=py312`, `line-length=88`, `preview=true`; `isort known-first-party=["accounts","config","grok_gateway","server","session_store","statsig","uploads"]`. Google-style docstrings on everything.
- Naming: `snake_case` funcs/vars, `CapWords` classes, `_leading_underscore` privates, `ALL_CAPS` constants (`MODEL_MODE_MAP`, TTLs); verbs `is_*`/`has_*`/`check_*`/`resolve_*`/`build_*`/`extract_*`/`persist_*`/`render_*`/`yield_*`; state holders `*_state`/`*_context`/`*_accumulator` (e.g. `SessionState`, `_ChatContext`, `_TurnState`).
- Typing: `type JsonValue = ...` alias, `TypedDict + Unpack` for turn kwargs (`TurnOptions`, `PickOptions`), `if TYPE_CHECKING:` for `AsyncIterator`/`Mapping`/`SqliteStore`.
- Async: `asyncio` everywhere — `curl-cffi AsyncSession` for REST, `websockets` async generators for turns/SSE (`AsyncIterator[dict|str]`), `asyncio.gather` for parallel downloads/uploads, per-session `asyncio.Lock` (`GrokSession.lock`) + global `SESSION_LOCK`, fire-and-forget via `asyncio.create_task` tracked in `_BACKGROUND_TASKS`. `SqliteStore` uses `threading.Lock` (sync SQLite under async callers); prune outside `SESSION_LOCK` to avoid blocking on disk I/O.
- Error handling: typed `GatewayError.kind` → `_fail_kind` → pool quarantine + `_gateway_http_status` → `HTTPException("grok error (kind): msg")`. Non-stream raises; mid-SSE logs or emits `response.failed`, never aborts silently. Explicit tuples + `contextlib.suppress` on close paths. Degraded detectors (placeholder `web_search` queries, fresh-render-on-edit with attachments + zero reasoning/text) abort as `kind=degraded` → failover, 502 if all accounts degraded.
- State: no DI framework. Module singletons: `store = SqliteStore(DB_PATH)`, `pool = AccountPool(...)`, `statsig = StatsigGenerator(store)`, `SESSIONS` dict. Per-request state in dataclasses, never globals. Add new endpoints in `server.py` (no routers package); add gateway frames in `grok_gateway.py:_apply_turn_event`; persist new state via `session_store.py`, not in-memory only.
- Example: new turn kwarg → extend `TurnOptions` TypedDict in `server.py`, thread via `Unpack[TurnOptions]` through `pick_account_and_*` → `stream_session_turn`.

## Important Files

| Path | Role |
|---|---|
| `server.py` | FastAPI `app`, all endpoints, translation/orchestration, SSE rendering |
| `grok_gateway.py` | Grok mgw WS client (`GrokSession`), Imagine WS, degraded detection |
| `accounts.py` | Round-robin `AccountPool`, cooldown/quarantine, hot-reload |
| `session_store.py` | SQLite persistence (sessions, accounts, caches) |
| `statsig.py` | Pure-Python `x-statsig-id` generator |
| `uploads.py` | Presigned upload + freeimage.host |
| `config.py` | Env config (`G2O_HOST/PORT/ACCOUNTS_FILE/DB_PATH/API_KEY/COOLDOWN/SESSION_TTL/MAX_SESSIONS/DEFAULT_MODEL/INCLUDE_SOURCES`) |
| `pyproject.toml` | Deps (`curl-cffi,fastapi,uvicorn,websockets,pydantic`), `requires-python>=3.12`, `ruff`+`ty` config |
| `.env.example` | Env template (`FREEIMAGE_API_KEY`, `G2O_*`) |
| `restart.sh` | Start/stop wrapper (`.server.pid`, `server.log`, `/healthz` wait) |
| `README.md` | Sole docs entry (quickstart, endpoints, models, accounts format) |
| `tools/capture_edit.py` | Debug: capture image-edit gateway frames |

## Runtime/Tooling Preferences

- Runtime: Python `>=3.12` only. No Bun/Node/TS toolchain (`package.json`, `tsconfig` absent).
- Package manager: `uv` canonical (`uv.lock` + `[dependency-groups] dev`); `requirements.txt` is pip fallback mirror (runtime deps only).
- Tooling: `ruff>=0.16.7` (lint+format), `ty>=0.0.78` (strict, `error-on-warning=true`). No pytest config — defaults apply. No Dockerfile/Compose, no `.github/` CI, no Makefile.
- Always use codebase-memory-mcp.

## Testing & QA

- Framework: stdlib `unittest` (`unittest.TestCase`, `IsolatedAsyncioTestCase` for async/SQLite), also runnable under `pytest>=9.1.1` (dev dep). No `conftest.py`, no coverage config/gates.
- Layout: `tests/test_degraded.py` (1219 lines, quarantine/failover), `test_include_sources.py` (appendix contract), `test_sqlite_persistence.py` (restart recovery), `test_images.py` / `test_attachment_upload.py` (upload semantics: fail-loud, partial success), `test_responses_input_text.py` (input flattening), `test_multi_turn_attachments.py` / `test_multiturn_memory.py` (cross-turn memory), `test_real_streaming.py` (SSE timing).
- Run:

```bash
python3 -m unittest discover -s tests -v   # full suite (or: pytest tests/ -v)
python3 -m unittest -v tests.test_degraded  # single file (or: pytest tests/test_degraded.py -v)
```

- Conventions: per-file `Run: python3 -m unittest -v tests.test_<name>` docstring + `unittest.main()` footer; self-contained mocks of gateway frames/uploads, no shared fixtures (`tests/__init__.py` is marker only). Failing upload-all = error assertion, never silent drop — preserve this invariant in new tests.
