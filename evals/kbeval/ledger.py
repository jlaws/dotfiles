"""Machine-local, locked spend ledger for eval runs.

One JSON object per line at `${XDG_STATE_HOME:-$HOME/.local/state}/dotfiles-kbeval/ledger.jsonl`.
Every append holds an exclusive `flock`, so concurrent runners never interleave or drop a line.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Dict, Iterator, Optional


def default_path() -> Path:
    """Return the ledger path under the XDG state directory."""
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "dotfiles-kbeval" / "ledger.jsonl"


def append(entry: dict, path: Optional[Path] = None) -> None:
    """Append one spend entry under an exclusive lock."""
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def entries(path: Optional[Path] = None) -> Iterator[dict]:
    """Yield every ledger entry, reading under a shared lock."""
    path = path or default_path()
    if not path.exists():
        return
    with open(path, encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        try:
            lines = handle.read().splitlines()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    for line in lines:
        if line.strip():
            yield json.loads(line)


def spent(period: str, path: Optional[Path] = None, purpose: Optional[str] = None) -> float:
    """Return total spend for a period, optionally for one purpose."""
    return sum(
        float(entry.get("costUsd") or 0.0)
        for entry in entries(path)
        if entry.get("period") == period and (purpose is None or entry.get("purpose") == purpose)
    )


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
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
