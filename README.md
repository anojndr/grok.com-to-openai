# grok-to-openai-api

Turn [grok.com](https://grok.com/) free accounts into an OpenAI-compatible API using FastAPI.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # edit: set PIXELVAULT_API_KEY and optionally G2O_API_KEY
python3 server.py     # serves on port 45080
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat Completions API (stream + non-stream) |
| `POST /v1/responses` | Responses API (stream + non-stream, `previous_response_id` chaining) |
| `GET /v1/models` | List available models |
| `GET /healthz` | Server health + account pool status |

## Models

- `grok-fast` — Grok 4.5 Fast (default)
- `grok-auto` — Auto mode
- `grok-expert` — Grok 4.5 Expert/Thinking
- `grok-heavy` — Grok 4.5 Heavy (requires paid tier)

## Features

### Multi-turn Conversations
Uses Grok's WebSocket Gateway (`wss://grok.com/ws/mgw/`) with persistent conversations.
Each follow-up attaches only the newest user message to an immutable user-message-chain
checkpoint, so replies to earlier messages start from the selected branch rather than
inheriting a later sibling turn.

### File Support
Attach images (data URLs, HTTP URLs), text files (inlined into prompt), and binary
files (uploaded via Grok's presigned upload pipeline). Supported input types:
- Chat Completions: `image_url`, `file` content parts
- Responses API: `input_image`, `input_file` items

If every attachment upload fails (e.g. anti-bot gating of grok.com REST), the
request fails with an upstream error instead of silently sending the prompt
without its files.

### Image Generation
Prompts matching image-generation intent ("generate an image of...", "draw...")
are routed to Grok's Imagine WebSocket (`wss://grok.com/ws/imagine/listen`).
Generated images are automatically uploaded to PixelVault and returned as
Markdown links in the assistant response.

### Load Balancing
Round-robin across all accounts in `accounts.txt`. Accounts that hit quota limits,
auth failures, or errors get cooldown periods before being retried. The file is
hot-reloaded on change (add/remove accounts without restarting).
### Degraded Accounts
Some grok accounts intermittently serve turns where the web-search tool runs
placeholder queries unrelated to the message ("current information and recent
sources", ...) and the model summarizes the unrelated results into word salad
that still looks like a successful response. `grok_gateway` detects this from
the streamed `tool_usage_card.web_search.args.query` events before the salad
renders, aborts the turn (`kind=degraded`), fails the request over to another
account, and quarantines the offending account for an hour. `/healthz` reports
the count as `degraded_accounts`. If every attempt lands on a degraded account
the request surfaces a 502 instead of garbage.

### Show Sources (llmcord-go)
To feed search citations into `llmcord-go`'s **Show Sources** button:
- Enable globally by setting `G2O_INCLUDE_SOURCES=1` (or `GROK_INCLUDE_SOURCES=1`) in `.env`.
- Or enable per request with `"include_sources": true` in the JSON request body (e.g. in `llmcord-go`, set `extra_body: {include_sources: true}`).

When enabled and Grok uses web search, an appendix formatted as:
```markdown
Sources
1. [Title](url) (domain) via `query`

Search Queries
1. `query`
```
is appended at the end of the turn. `llmcord-go` automatically hides this appendix from the visible message while rendering and parses it to populate the **Show Sources** button and paginated sources view.

## Configuration (.env)

```ini
PIXELVAULT_API_KEY=pv_live_...   # Required for image hosting
G2O_API_KEY=                     # Optional: protect this API
G2O_PORT=45080                   # Default port
G2O_ACCOUNTS_FILE=accounts.txt   # Cookie file path
G2O_COOLDOWN=300                 # Failure cooldown seconds
G2O_SESSION_TTL=3600             # Multi-turn session TTL seconds
G2O_INCLUDE_SOURCES=0            # Optional: enable Show Sources bridge appendix (0 or 1)
```

## Accounts Format

`accounts.txt` contains Netscape cookie blocks separated by optional headers:

```
account 1:

# Netscape HTTP Cookie File
.grok.com	TRUE	/	TRUE	1234567890	sso	eyJ...
.grok.com	TRUE	/	TRUE	1234567890	sso-rw	eyJ...
...

account 2:
...
```

Minimum required: `sso` cookie. Recommended: `sso-rw`, `x-userid`, `cf_clearance`.

## Usage Examples

```bash
# Simple chat
curl http://localhost:45080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-fast","messages":[{"role":"user","content":"Hello!"}]}'

# Streaming
curl http://localhost:45080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-fast","stream":true,"messages":[{"role":"user","content":"Hi"}]}'

# Multi-turn
curl http://localhost:45080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-fast","messages":[
    {"role":"user","content":"My name is Alice"},
    {"role":"assistant","content":"Nice to meet you!"},
    {"role":"user","content":"What is my name?"}
  ]}'

# Responses API with chaining
curl http://localhost:45080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-fast","input":"What is 2+2?"}'
# Then use the response ID:
curl http://localhost:45080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-fast","previous_response_id":"resp_xxx","input":"Now multiply that by 3"}'

# Image generation
curl http://localhost:45080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-fast","messages":[{"role":"user","content":"generate an image of a cat"}]}'
```

## Architecture

```
server.py          FastAPI app, OpenAI-compatible endpoints
accounts.py        Account pool manager (round-robin, cooldown, hot-reload)
session_store.py   SQLite persistence for multi-turn sessions, accounts, and caches
statsig.py         x-statsig-id generator (pure Python, no browser needed)
grok_gateway.py    WebSocket Gateway client (chat sessions)
uploads.py         File upload v2 + PixelVault integration
config.py          Environment configuration
```

## Notes

- Free accounts have ~7 queries/hour for chat and limited image generations.
- The server automatically retries on different accounts if one hits a limit.
- Rate-limited accounts enter cooldown and are skipped until the window resets.
- Accounts caught serving degraded (word-salad) responses are quarantined for
  an hour; see Degraded Accounts above.
- The `x-statsig-id` anti-bot header is generated locally in pure Python
  (no browser or JS engine required).
