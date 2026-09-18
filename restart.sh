#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$DIR/.server.pid"
LOG_FILE="$DIR/server.log"

# Load PORT from env or .env file
if [ -z "${G2O_PORT:-}" ] && [ -f "$DIR/.env" ]; then
    ENV_PORT=$(grep -E '^[[:space:]]*G2O_PORT=' "$DIR/.env" | cut -d= -f2 | tr -d '"'\'' ' || true)
    if [ -n "$ENV_PORT" ]; then
        G2O_PORT="$ENV_PORT"
    fi
fi
PORT="${G2O_PORT:-45080}"
BASE_URL="http://localhost:$PORT"

# Locate Python executable: prefer THIS repo's .venv so an activated
# VIRTUAL_ENV from a sibling bridge never hijacks the interpreter.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x "$DIR/.venv/bin/python3" ]; then
        PYTHON="$DIR/.venv/bin/python3"
    elif [ -x "$DIR/.venv/bin/python" ]; then
        PYTHON="$DIR/.venv/bin/python"
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then
        PYTHON="$VIRTUAL_ENV/bin/python3"
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
        PYTHON="$VIRTUAL_ENV/bin/python"
    elif command -v python3 &>/dev/null; then
        PYTHON="$(command -v python3)"
    elif command -v python &>/dev/null; then
        PYTHON="$(command -v python)"
    fi
fi

if [ -z "${PYTHON:-}" ] || ! "$PYTHON" -c "import uvicorn" &>/dev/null; then
    echo "✗ Python interpreter (${PYTHON:-none}) does not have uvicorn installed — refusing to restart." >&2
    echo "  Install dependencies with 'uv sync' or 'pip install -r requirements.txt'." >&2
    exit 1
fi

# ── Stop if already running ──────────────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        # Check command line to avoid killing unrelated processes on PID reuse;
        # scoped to $DIR so a stale pid pointing at a sibling bridge is never killed.
        CMDLINE=""
        if [ -f "/proc/$OLD_PID/cmdline" ]; then
            CMDLINE=$(tr '\0' ' ' < "/proc/$OLD_PID/cmdline" 2>/dev/null || true)
        fi
        if [ -z "$CMDLINE" ] || echo "$CMDLINE" | grep -qF "$DIR"; then
            echo "Stopping server (pid $OLD_PID)..."
            kill "$OLD_PID" 2>/dev/null || true
            for _ in $(seq 1 20); do
                kill -0 "$OLD_PID" 2>/dev/null || break
                sleep 0.5
            done
            kill -9 "$OLD_PID" 2>/dev/null || true
        fi
    fi
    rm -f "$PID_FILE"
fi

# also kill anything bound to the port
if command -v fuser &>/dev/null; then
    fuser -k "$PORT"/tcp 2>/dev/null || true
fi

# Wait until the port is actually free. Starting uvicorn while the old
# process still holds the port fails with "[Errno 98] address already in
# use" and leaves a stale .server.pid behind with nothing listening.
for _ in $(seq 1 30); do
    if command -v ss &>/dev/null; then
        ss -tln 2>/dev/null | grep -qE "[:.]$PORT([[:space:]]|$)" || break
    else
        (echo > /dev/tcp/127.0.0.1/"$PORT") 2>/dev/null || break
    fi
    sleep 0.5
done
if command -v ss &>/dev/null; then
    if ss -tln 2>/dev/null | grep -qE "[:.]$PORT([[:space:]]|$)"; then
        echo "✗ Port $PORT still bound after stop — refusing to start a doomed process. Check $LOG_FILE" >&2
        exit 1
    fi
elif (echo > /dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
    echo "✗ Port $PORT still bound after stop — refusing to start a doomed process. Check $LOG_FILE" >&2
    exit 1
fi

 # ── Start ────────────────────────────────────────────────────────────────────
 echo "Starting server on port $PORT..."
 cd "$DIR"
if command -v setsid &>/dev/null; then
    setsid nohup "$PYTHON" -m uvicorn server:app --host 0.0.0.0 --port "$PORT" \
        < /dev/null >> "$LOG_FILE" 2>&1 &
else
    nohup "$PYTHON" -m uvicorn server:app --host 0.0.0.0 --port "$PORT" \
        < /dev/null >> "$LOG_FILE" 2>&1 &
fi
START_PID=$!
echo "$START_PID" > "$PID_FILE"
 
 # ── Wait for readiness ───────────────────────────────────────────────────────
 for _ in $(seq 1 30); do
     if curl -sf "$BASE_URL/healthz" >/dev/null 2>&1; then
         break
     fi
     sleep 0.5
 done
 
if ! curl -sf "$BASE_URL/healthz" >/dev/null 2>&1; then
    echo "✗ Server did not become ready — check $LOG_FILE" >&2
    tail -n 20 "$LOG_FILE" >&2 || true
    kill "$START_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$START_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    exit 1
fi
echo "✓ Server is up: $BASE_URL"

# $! can be a wrapper subshell (nohup+redirect double-fork), not the uvicorn
# process itself. Record the pid that actually holds the port so the pid
# file never points at a dead process while the server runs (or vice versa).
# Scoped to $DIR so a sibling bridge holding the port pattern never lands in .server.pid.
if command -v ss &>/dev/null; then
    LISTENER_PID=$(ss -tlnp 2>/dev/null | grep -E "[:.]$PORT([[:space:]]|$)" | grep -oE 'pid=[0-9]+' | head -n 1 | cut -d= -f2 || true)
    if [ -n "${LISTENER_PID:-}" ] && kill -0 "$LISTENER_PID" 2>/dev/null; then
        LISTENER_CMD=$(tr '\0' ' ' < "/proc/$LISTENER_PID/cmdline" 2>/dev/null || true)
        if echo "$LISTENER_CMD" | grep -qF "$DIR"; then
            echo "$LISTENER_PID" > "$PID_FILE"
        fi
    fi
fi
if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "✗ Pid file points at a dead process — check $LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
fi
echo ""
echo "─────────────────────────────────────────────"
echo "  Base URL:  $BASE_URL"
echo "  API:       $BASE_URL/v1/chat/completions"
echo "             $BASE_URL/v1/responses"
echo "  Models:    $BASE_URL/v1/models"
echo "  Health:    $BASE_URL/healthz"
echo "  Log:       tail -f $LOG_FILE"
echo "─────────────────────────────────────────────"

TAIL_LOG=1
if [ ! -t 1 ] || [ "${1:-}" = "--no-tail" ] || [ "${1:-}" = "--background" ] || [ "${1:-}" = "-b" ]; then
    TAIL_LOG=0
fi
if [ "${1:-}" = "--tail" ] || [ "${1:-}" = "-f" ]; then
    TAIL_LOG=1
fi
if [ "$TAIL_LOG" -eq 1 ]; then
    echo ""
    echo "Tailing log (Ctrl+C to stop watching, server keeps running):"
    echo ""
    tail -f "$LOG_FILE"
fi
