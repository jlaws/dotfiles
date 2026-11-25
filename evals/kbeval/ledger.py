"""Machine-local, locked spend ledger for eval runs.

One JSON object per line at `${XDG_STATE_HOME:-$HOME/.local/state}/dotfiles-kbeval/ledger.jsonl`.
Every write holds an exclusive `flock`. A batch first reserves its cap with `reserve`, which checks
the budget and appends under one lock, so concurrent runners cannot both spend the same headroom;
the batch then settles the reservation to its actual cost. A crash leaves the reservation charged.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional


class LedgerError(Exception):
    """Raised when the ledger holds a line that cannot be trusted."""


def default_path() -> Path:
    """Return the ledger path under the XDG state directory."""
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "dotfiles-kbeval" / "ledger.jsonl"


def _validate(entry: dict) -> None:
    cost = entry.get("costUsd", 0.0)
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost):
        raise LedgerError(f"costUsd must be a finite number, not {cost!r}")
    if cost < 0 and entry.get("charged") != "settle":
        raise LedgerError(f"only a settle entry may be negative, not {entry.get('charged')!r}")


def _parse(lines: List[str], path: Path) -> List[dict]:
    parsed = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            _validate(entry)
        except (ValueError, LedgerError) as error:
            raise LedgerError(
                f"{path}:{number}: unreadable ledger line ({error}); fix it by hand"
            ) from error
        parsed.append(entry)
    return parsed


def _line(entry: dict) -> str:
    _validate(entry)
    return json.dumps(entry, sort_keys=True) + "\n"


def append(entry: dict, path: Optional[Path] = None) -> None:
    """Append one spend entry under an exclusive lock."""
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _line(entry)
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def reserve(entry: dict, ceiling: float, headroom: float, path: Optional[Path] = None) -> bool:
    """Append `entry` only if the period's spend plus its cost plus `headroom` fits the ceiling.

    The check and the append happen under one exclusive lock.
    """
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _line(entry)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            current = _total(_parse(handle.read().splitlines(), path), entry["period"])
            if current + float(entry["costUsd"]) + headroom > ceiling + 1e-9:
                return False
            handle.seek(0, os.SEEK_END)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
            return True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def entries(path: Optional[Path] = None) -> Iterator[dict]:
    """Yield every ledger entry, reading under a shared lock; a bad line raises LedgerError."""
    path = path or default_path()
    if not path.exists():
        return
    with open(path, encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        try:
            lines = handle.read().splitlines()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    yield from _parse(lines, path)


def _total(items: List[dict], period: str, purpose: Optional[str] = None) -> float:
    return sum(
        float(entry.get("costUsd") or 0.0)
        for entry in items
        if entry.get("period") == period and (purpose is None or entry.get("purpose") == purpose)
    )


def spent(period: str, path: Optional[Path] = None, purpose: Optional[str] = None) -> float:
    """Return total spend for a period, optionally for one purpose."""
    return _total(list(entries(path)), period, purpose)


def snapshot(out: Path, period: str, path: Optional[Path] = None) -> None:
    """Write the period's totals, overall and by purpose and model, to `out`."""
    total = 0.0
    by_purpose: Dict[str, float] = {}
    by_model: Dict[str, float] = {}
    for entry in entries(path):
        if entry.get("period") != period:
            continue
        cost = float(entry.get("costUsd") or 0.0)
        total += cost
        purpose = entry.get("purpose", "unknown")
        model = entry.get("model", "unknown")
        by_purpose[purpose] = round(by_purpose.get(purpose, 0.0) + cost, 4)
        by_model[model] = round(by_model.get(model, 0.0) + cost, 4)
    doc = {
        "period": period,
        "totals": {"totalUsd": round(total, 4), "byPurpose": by_purpose, "byModel": by_model},
    }
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
