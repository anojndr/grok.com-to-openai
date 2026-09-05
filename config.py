"""Configuration via environment variables (no secrets hardcoded)."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    """Tiny .env loader — must run before any os.getenv below."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv()

HOST = os.getenv("G2O_HOST", "0.0.0.0")
PORT = int(os.getenv("G2O_PORT", "45080"))

ACCOUNTS_FILE = os.getenv("G2O_ACCOUNTS_FILE", str(BASE_DIR / "accounts.txt"))
# SQLite database path for persistent sessions, accounts and caches
DB_PATH = os.getenv("G2O_DB_PATH", str(BASE_DIR / "data" / "grok_store.db"))

# Optional API key to protect THIS server (clients must send `Authorization: Bearer <key>`).
API_KEY = os.getenv("G2O_API_KEY", "")

# FreeImage.host API key for hosting generated images (never hardcode; set in .env)
# Get a key at https://freeimage.host/page/api after signing up.
FREEIMAGE_API_KEY = os.getenv("FREEIMAGE_API_KEY", "")
FREEIMAGE_BASE = os.getenv("FREEIMAGE_BASE", "https://freeimage.host")

GROK_BASE = os.getenv("G2O_GROK_BASE", "https://grok.com")

# Browser identity used for all upstream calls (keep consistent per deployment).
USER_AGENT = os.getenv(
    "G2O_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
)

# Cooldown applied after auth/quota failures (seconds)
COOLDOWN_SECONDS = int(os.getenv("G2O_COOLDOWN", "300"))
# Session idle TTL for multi-turn conversation reuse (seconds)
SESSION_TTL = int(os.getenv("G2O_SESSION_TTL", "3600"))
# Max concurrent gateway sessions
MAX_SESSIONS = int(os.getenv("G2O_MAX_SESSIONS", "64"))

DEFAULT_MODEL = os.getenv("G2O_DEFAULT_MODEL", "grok-fast")
# When enabled (1/true/yes/on), append a "Sources" + "Search Queries" appendix
# to answers that have Grok web search results for llmcord-go's "Show Sources" button.
_INCLUDE_SOURCES_RAW = os.getenv("G2O_INCLUDE_SOURCES")
if _INCLUDE_SOURCES_RAW is None:
    _INCLUDE_SOURCES_RAW = os.getenv("GROK_INCLUDE_SOURCES", "0")
INCLUDE_SOURCES = _INCLUDE_SOURCES_RAW.strip().lower() in ("1", "true", "yes", "on")
