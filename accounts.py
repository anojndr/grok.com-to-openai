"""Account pool: parse accounts.txt, hot-reload, round-robin load balancing.

accounts.txt format: blocks of Netscape cookie files separated by optional
`account N:` headers. Each block must contain an `sso` cookie line. The
`x-userid` cookie (if present) carries the gateway user id.
"""

from __future__ import annotations

import asyncio
import itertools
import re
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

BLOCK_SPLIT = re.compile(r"(?m)^account\s+\d+:\s*$")
INTERESTING_COOKIES = {
    "sso",
    "sso-rw",
    "cf_clearance",
    "__cf_bm",
    "x-userid",
    "grok_device_id",
}


@dataclass
class Account:
    index: int
    cookies: dict[str, str]
    user_id: str = ""

    @property
    def key(self) -> str:
        uid = self.cookies.get("x-userid", "")
        if uid:
            return "u:" + uid
        return (
            "s:" + hashlib.sha256(self.cookies.get("sso", "").encode()).hexdigest()[:24]
        )

    cooldown_until: float = 0.0
    degraded_until: float = 0.0
    in_flight: int = 0
    total_requests: int = 0
    failed_requests: int = 0

    @property
    def degraded(self) -> bool:
        return time.time() < self.degraded_until

    @property
    def sso(self) -> str:
        return self.cookies.get("sso", "")

    def cookie_header(self, extra: dict[str, str] | None = None) -> str:
        jar = {k: v for k, v in self.cookies.items() if k in INTERESTING_COOKIES}
        if extra:
            jar.update(extra)
        return "; ".join(f"{k}={v}" for k, v in jar.items())

    def available(self) -> bool:
        return (
            bool(self.sso)
            and time.time() >= self.cooldown_until
            and time.time() >= self.degraded_until
        )

    def mark_failed(self, cooldown: float) -> None:
        self.failed_requests += 1
        self.cooldown_until = max(self.cooldown_until, time.time() + cooldown)


class AccountPool:
    def __init__(self, path: str | Path, cooldown_seconds: int = 300, store=None):
        self.path = Path(path)
        self.cooldown_seconds = cooldown_seconds
        self.store = store
        self._accounts: list[Account] = []
        self._mtime: float = 0.0
        self._rr = itertools.count()
        self._lock = asyncio.Lock()

    async def reload_if_changed(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime == self._mtime and self._accounts:
            return
        async with self._lock:
            if mtime == self._mtime and self._accounts:
                return
            new_accounts = self._parse(self.path)
            old_by_key = {a.key: a for a in self._accounts}
            stored_states = self.store.get_all_account_states() if self.store else {}
            for acc in new_accounts:
                old = old_by_key.get(acc.key)
                if old is not None:
                    acc.cooldown_until = old.cooldown_until
                    acc.degraded_until = old.degraded_until
                    acc.in_flight = old.in_flight
                    acc.total_requests = old.total_requests
                    acc.failed_requests = old.failed_requests
                elif acc.key in stored_states:
                    st = stored_states[acc.key]
                    acc.cooldown_until = st.get("cooldown_until", 0.0)
                    acc.degraded_until = st.get("degraded_until", 0.0)
                    acc.total_requests = st.get("total_requests", 0)
                    acc.failed_requests = st.get("failed_requests", 0)
            self._accounts = new_accounts
            self._mtime = mtime

    @staticmethod
    def _parse(path: Path) -> list[Account]:
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8", errors="replace")
        blocks = BLOCK_SPLIT.split(raw)
        if not any("sso" in b for b in blocks):
            # maybe a single cookie file without headers
            blocks = [raw]
        accounts: list[Account] = []
        n = 0
        for block in blocks:
            block = block.strip()
            if not block or "sso" not in block and "\t" not in block:
                continue
            cookies: dict[str, str] = {}
            for line in block.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 7:
                    name, value = parts[5].strip(), parts[6].strip()
                    if name in INTERESTING_COOKIES:
                        cookies[name] = value
            if not cookies.get("sso"):
                continue
            n += 1
            accounts.append(
                Account(
                    index=n,
                    cookies=cookies,
                    user_id=cookies.get("x-userid", ""),
                )
            )
        return accounts

    def snapshot(self) -> list[Account]:
        return list(self._accounts)

    def acquire(
        self, exclude: set[str] | None = None, include_degraded: bool = False
    ) -> Account | None:
        """Pick the next account to serve a request.

        Round-robins over available accounts, skipping every key in
        `exclude` so a failover request can hand each attempt a distinct
        account. When none are available, the soonest-recoverable cooling
        account is used; degraded-quarantined accounts only join that
        last-resort pool when `include_degraded` is set (turn failover sets
        it so every account is tried before surfacing an error — the
        degraded-turn detector still aborts a bad turn before it completes).
        """
        exclude = exclude or set()
        candidates = [a for a in self._accounts if a.sso and a.key not in exclude]
        if not candidates:
            return None
        available = [a for a in candidates if a.available()]
        if available:
            acc = available[next(self._rr) % len(available)]
        else:
            now = time.time()
            cooling = [a for a in candidates if now >= a.degraded_until]
            if not cooling and include_degraded:
                cooling = candidates
            if not cooling:
                return None
            acc = min(
                cooling,
                key=lambda a: (max(a.cooldown_until, a.degraded_until), a.index),
            )
        acc.total_requests += 1
        if self.store:
            self.store.save_account_state(
                acc.key,
                acc.cooldown_until,
                acc.degraded_until,
                acc.total_requests,
                acc.failed_requests,
            )
        return acc

    def acquire_by_key(self, key: str) -> Account | None:
        acc = next((a for a in self._accounts if a.key == key), None)
        if acc and acc.available():
            acc.total_requests += 1
            if self.store:
                self.store.save_account_state(
                    acc.key,
                    acc.cooldown_until,
                    acc.degraded_until,
                    acc.total_requests,
                    acc.failed_requests,
                )
            return acc
        return None

    def release_ok(self, acc: Account) -> None:
        acc.cooldown_until = 0.0
        if self.store:
            self.store.save_account_state(
                acc.key,
                acc.cooldown_until,
                acc.degraded_until,
                acc.total_requests,
                acc.failed_requests,
            )

    def release_fail(self, acc: Account, kind: str = "generic") -> None:
        cooldown = self.cooldown_seconds
        if kind == "auth":
            cooldown = max(cooldown, 1800)
        elif kind == "quota":
            # retry after the top-of-hour window typically used by grok
            now = time.time()
            until_next_hour = 3600 - (now % 3600) + 60
            cooldown = max(cooldown, min(until_next_hour, 3600))
        elif kind == "degraded":
            # Degraded gateways serve word salad as successful turns; quarantine
            # the account for an hour so real traffic stops routing through it.
            # release_ok must not clear this deadline; after it lapses the
            # account is re-admitted and the first turn re-detects degradation.
            cooldown = max(cooldown, 3600)
            acc.degraded_until = max(acc.degraded_until, time.time() + cooldown)
        acc.mark_failed(cooldown)
        if self.store:
            self.store.save_account_state(
                acc.key,
                acc.cooldown_until,
                acc.degraded_until,
                acc.total_requests,
                acc.failed_requests,
            )
