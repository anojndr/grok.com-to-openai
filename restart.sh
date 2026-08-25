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

# ── Stop if already running ──────────────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        # Check command line to avoid killing unrelated processes on PID reuse
        CMDLINE=""
        if [ -f "/proc/$OLD_PID/cmdline" ]; then
            CMDLINE=$(tr '\0' ' ' < "/proc/$OLD_PID/cmdline" 2>/dev/null || true)
        fi
        if [ -z "$CMDLINE" ] || echo "$CMDLINE" | grep -qE "uvicorn|server:app"; then
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
    sleep 0.5
fi

# ── Start ────────────────────────────────────────────────────────────────────
echo "Starting server on port $PORT..."
cd "$DIR"
nohup python3 -m uvicorn server:app --host 0.0.0.0 --port "$PORT" \
    >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

# ── Wait for readiness ───────────────────────────────────────────────────────
for _ in $(seq 1 30); do
    if curl -sf "$BASE_URL/healthz" >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

if curl -sf "$BASE_URL/healthz" >/dev/null 2>&1; then
    echo "✓ Server is up: $BASE_URL"
else
    echo "✗ Server may not be ready yet — check $LOG_FILE"
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
echo ""
echo "Tailing log (Ctrl+C to stop watching, server keeps running):"
echo ""
tail -f "$LOG_FILE"
