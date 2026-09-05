# Repository Guidelines

## Project Overview

OpenAI-compatible FastAPI gateway over grok.com free accounts (`wss://grok.com/ws/mgw`).
Exposes `POST /v1/chat/completions`, `POST /v1/responses`, `GET /v1/models`, `GET /healthz`.
Models: `grok-fast` (default), `grok-auto`, `grok-expert`, `grok-heavy` via `MODEL_MODE_MAP`.

## Architecture & Data Flow

Single-process FastAPI app, flat top-level modules, no `src/` or DI framework.
Module singletons in `server.py:69-73` (`store`, `pool`, `statsig`, `SESSIONS`, `SESSION_LOCK`); collaborators passed explicitly.

Chat flow (`server.py:1537`):
`check_auth` (optional `G2O_API_KEY` Bearer) → `resolve_mode(model)` (unknown → `fast`) → `user_texts`/`content_to_text` flatten → `extract_attachments` (text vs file/image jobs) → `pick_account_and_stream_turn` (`server.py:1020`) → per-attempt `pool.acquire(exclude=tried)` → `get_or_create_session` → `fork_session_state` (moves live `ws` to fork, `server.py:119`) → `stream_session_turn` (`server.py:744`, uploads via `uploads.upload_file` with current account) → `GrokSession.ask()` (`grok_gateway.py:558`) → OpenAI SSE (`data: {...}` + `data: [DONE]`) or JSON.
`POST /v1/responses` (`server.py:2035`) reuses same core with `previous_response_id` chaining.

Gateway turn: `asyncio.Lock` per `GrokSession`; reconnect if `not alive() or ws_mode != model_mode`; send `conversation.item.create` + `response.create`; `ws.recv()` loop (`idle_timeout=120s`, `max_turn_timeout=300s`); abort degraded turns via `unrelated_queries()` + fast-mode placeholder heuristics.

State (two tiers):
- Memory: `SESSIONS: dict[str, SessionState]` (`server.py:91`), `SESSION_LOCK`-guarded, pruned by `SESSION_TTL=3600s` / `MAX_SESSIONS=64`.
- SQLite `SqliteStore` (`data/grok_store.db`, WAL): `sessions` / `account_states` / `uid_cache` / `statsig_cache`; attachment cap `MAX_TRACKED_ATTACHMENTS=12`.

Failover: walk whole `AccountPool`, `release_fail(kind)` tiers (generic 300s, auth ≥1800s, quota → top-of-hour, degraded → 1h quarantine); commit stream on first yielded event.

## Key Directories

Flat root, no `src/`, `docs/`, `scripts/`, `examples/`:
- `server.py` (2766L): app, 4 routes, session map, turn orchestration, SSE formatting.
- `grok_gateway.py` (823L): `GrokSession`, `TurnResult`, `GatewayError`, `RenderFilter`.
- `accounts.py`: `Account` + `AccountPool` (cookie-block parse, hot-reload, round-robin).
- `session_store.py`: `SqliteStore` + `clean_attachment_rows`.
- `uploads.py`: Grok v2 presigned upload (`init→PUT→complete→poll`), `freeimage_upload[_from_url]` (freeimage.host Chevereto v1), SSRF guard.
- `statsig.py`: `StatsigGenerator` forging `x-statsig-id`, graceful-absent fallback.
- `config.py`: `G2O_*` env source of truth.
- `tests/` (9 files), `tools/capture_edit.py` (live-account debug probe, not runtime).

## Development Commands

```bash
pip install -r requirements.txt   # documented quick-start
uv sync                           # preferred when using uv (uv.lock present)
cp .env.example .env              # then set FREEIMAGE_API_KEY, optional G2O_API_KEY
python3 server.py                 # serve on $G2O_PORT (default 45080)
python3 -m uvicorn server:app --host 0.0.0.0 --port "$PORT"
./restart.sh                      # kill .server.pid + fuser PORT, start nohup uvicorn, wait /healthz, tail server.log
curl -sf http://localhost:$PORT/healthz
python3 tools/capture_edit.py [n] # live grok probe; burns quota, debug only
```

## Code Conventions & Common Patterns

- `snake_case` everywhere; `from __future__ import annotations` in core modules; stdlib → third-party (`fastapi`/`curl_cffi`/`websockets`) → local; heavy deps imported function-locally.
- Naming: `*_key` = identity (`session_key`/`account_key`/`chain_key=sha256({auth,users})[:24]`), `*_prompt` = gateway-bound text vs `users` = raw texts; `_`-prefix = private (`_uid_for`, `_sig_headers`).
- Async: `asyncio`-native; per-session `asyncio.Lock` spans whole turn; global `SESSION_LOCK` covers map only (never network/SQLite); `asyncio.wait_for` timeouts; `AsyncIterator` event streams (`{type, ...}` dicts).
- Errors as narrow kinds, never blind `except Exception`: `GatewayError(kind,msg)` → `HTTPException(429 if quota else 502/503)` after exhausting accounts; parse guards catch `(ValueError, TypeError, AttributeError)` (+`RuntimeError` where the callee raises it, e.g. statsig `generate`); swallowed paths log at `debug` (`uvicorn.error`); socket/file `close()` cleanup uses `contextlib.suppress(OSError, RuntimeError, AttributeError)`; `UploadError` → drop bad data-URL or raise upstream; `400` non-string prompts, `401` bad key; per-frame narrow `except` + `debug` log, semantic abort at turn level; `release_fail(kind)` tiers (generic 300s, auth ≥1800s, quota → top-of-hour, degraded → 1h quarantine).
- Dataclasses (`Account`/`SessionState`/`TurnResult`); sha256 attachment dedup; `RenderFilter` strips `<grok:render>` incrementally; long comments on socket hand-off, transcript resend, attachment re-mention.
- Format: `ruff` line-length 88, `target-version = py312`; `ty` strict (`all = "error"`, `error-on-warning = true`).

Example turn driver:
```python
async for event in stream_session_turn(sess, prompt, history_prompt, ...):
    # event: {"type": "text_delta"|"reasoning_delta"|"image_url"|"done", ...}
```

## Important Files

| Path | Role |
|------|------|
| `server.py:69,2763` | `app` entrypoint; `__main__` runs `uvicorn.run(app, host=config.HOST, port=config.PORT)` |
| `server.py:1537,2035,2739,2750` | `chat_completions`, `responses_api`, `models`, `healthz` routes |
| `server.py:1020,1395,744,119,174` | `pick_account_and_stream_turn`, `pick_account_and_turn`, `stream_session_turn`, `fork_session_state`, `get_or_create_session` |
| `grok_gateway.py:418,452,558` | `GrokSession`, `connect`, `ask` |
| `config.py` | Tiny `.env` loader + `HOST`/`PORT`/`DB_PATH`/`GROK_BASE`/`USER_AGENT`/cooldowns |
| `accounts.txt` | Netscape cookie blocks (`account N:` headers; min `sso`, want `sso-rw,x-userid,cf_clearance`); hot-reloaded, gitignored |
| `restart.sh` | Dev restart loop (PID + `fuser -k PORT/tcp`, `nohup uvicorn`, `/healthz` poll) |
| `README.md` | Sole spec: endpoints, models, `.env` table, curl examples, module map |

## Runtime/Tooling Preferences

- Runtime: Python `>=3.12` only (`pyproject.toml`, `tool.ruff target-version = py312`, `tool.ty.environment python-version = 3.12`). No Node/Bun/npm, Docker, Compose, CI, Makefile.
- Package manager: `uv` preferred (`uv.lock` pinned, `[tool.uv] package = false` — service, not wheel); `pip install -r requirements.txt` fallback (5 unpinned deps: `curl_cffi, fastapi, uvicorn, websockets, pydantic`).
- Server: `server:app` on uvicorn; port resolves `G2O_PORT` env → `.env` → `45080`.
- Config: `config.py` + `.env` (`G2O_HOST/PORT/ACCOUNTS_FILE/DB_PATH/API_KEY/COOLDOWN/SESSION_TTL/MAX_SESSIONS/INCLUDE_SOURCES`, `FREEIMAGE_API_KEY/BASE`). Never commit `.env`/`accounts.txt` (see `.gitignore`); read credential structure only, never print keys.
- Strict tooling (always on, no exceptions):
  > Always use `https://docs.astral.sh/uv/` with everything enabled and `https://docs.astral.sh/ty/` with everything enabled, then fix all of the issues. Make sure to actually fix all of the issues instead of suppressing them.
  - That means: `ty` with `[tool.ty.rules] all = "error"` + strict analysis + `error-on-warning`; `uv check` (ty), `uv format --check` (ruff format), `uv audit`, and `ruff check .` must ALL be clean before yielding.
  - Fix, don't suppress: narrow `except Exception` to the real failure types + log (`logging.getLogger("uvicorn.error").debug/...`), use `contextlib.suppress(<narrow>)` only for close/cleanup races, never add `noqa` / `ty: ignore` / `type: ignore` / per-rule `ignore` to silence a finding.
  - Commands: `ty check`, `uv check`, `uv format --check` (or `uv format` to apply), `uv audit`, `uvx ruff check .`.

## Testing & QA

- Framework: stdlib `unittest` only (`unittest.IsolatedAsyncioTestCase` + `unittest.mock.AsyncMock/patch`); no pytest, `conftest.py`, coverage gates, or CI. Each `tests/test_*.py` ends with `unittest.main()`.
- Run:
```bash
python3 -m unittest discover -s tests -v
python3 -m unittest -v tests.test_degraded
python3 -m unittest -v tests.test_degraded.DegradedDetectionTests.test_flags_placeholder_queries
python3 tests/test_images.py -v
```
- Style: fully offline fakes — `FakeWS` scripting gateway frames (`tests/test_degraded.py:40+`), temp `accounts.txt` + `pool.reload_if_changed()`, `FakeRequest` over ASGI scope (`tests/test_sqlite_persistence.py:25-52`); temp SQLite via `tempfile.mkdtemp()` + manual `shutil.rmtree` in `tearDown`. `test_real_streaming.py` is mocked despite name; run sparingly if hitting live accounts.
- Lint/typecheck: `ruff check .`, `ty check` (strict, error-on-warning). Keep both green: combine nested `with` (SIM117), `_`-prefix unused unpacks (RUF059), `ClassVar` for mutable class constants (RUF012), `await asyncio.to_thread(...)` for blocking IO in async tests (ASYNC230), explicit `[DONE]`/empty guards instead of swallowing SSE parse errors.
