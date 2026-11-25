"""Machine-local, locked spend ledger for eval runs.

One JSON object per line at `${XDG_STATE_HOME:-$HOME/.local/state}/dotfiles-kbeval/ledger.jsonl`.
Every write holds an exclusive `flock`. A batch first reserves its cap with `reserve`, which checks
the budget and appends under one lock, so concurrent runners cannot both spend the same headroom;
the batch then settles the reservation to its actual cost. A crash leaves the reservation charged.

A reserve row carries an `id` (and `replaces`, the earlier reservation of the same batch it
follows); a settle row carries `settles`, the `id` it closes. `open_reservations` lists the ones
no settle row names. `settle_open` closes one by hand at a stated real cost, the only way out of
an interrupted run (with `reservation_id` when a rerun left several open), because the cost of runs in flight is unknown and is never released for you.
`settle_id` appends a settle row only if none already names that reservation.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple


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


def _append_if(
    path: Optional[Path], decide: Callable[[List[dict]], Optional[dict]]
) -> Optional[dict]:
    """Read the ledger, let `decide` pick an entry, and append it, all under one exclusive lock.

    `decide` returns the entry to append or None to append nothing; an exception it raises
    propagates after the lock is released. Returns what was appended.
    """
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            entry = decide(_parse(handle.read().splitlines(), path))
            if entry is None:
                return None
            line = _line(entry)
            handle.seek(0, os.SEEK_END)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
            return entry
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _key(entry: dict) -> Tuple[Optional[str], ...]:
    return tuple(entry.get(f) for f in ("purpose", "staging", "model", "suite", "batch"))


def reserve(
    entry: dict, ceiling: float, headroom: float, path: Optional[Path] = None
) -> Optional[dict]:
    """Append `entry` only if the period's spend plus its cost plus `headroom` fits the ceiling.

    The check and the append happen under one exclusive lock. Returns the appended entry, which
    gains an `id` and, when an earlier reservation in the period has the same purpose, staging,
    model, suite, and batch, `replaces` naming that row; returns None when it does not fit.
    """

    def decide(items: List[dict]) -> Optional[dict]:
        period = entry["period"]
        if _total(items, period) + float(entry["costUsd"]) + headroom > ceiling + 1e-9:
            return None
        row = {**entry, "id": uuid.uuid4().hex}
        earlier = [
            e
            for e in items
            if e.get("period") == period and e.get("charged") == "reserved" and _key(e) == _key(row)
        ]
        if earlier:
            row["replaces"] = earlier[-1].get("id") or earlier[-1].get("ts")
        return row

    return _append_if(path, decide)


def _open(items: List[dict], period: str) -> List[dict]:
    """Return the period's reservations that no settle row closes, in ledger order."""
    named = {e["settles"] for e in items if e.get("charged") == "settle" and e.get("settles")}
    closed: set = set()
    pending: Dict[Tuple[Optional[str], ...], List[int]] = {}
    rows = [e for e in items if e.get("period") == period]
    for index, row in enumerate(rows):
        if row.get("charged") == "reserved":
            if row.get("id") in named or (not row.get("id") and row.get("ts") in named):
                closed.add(index)
            elif not row.get("id"):
                pending.setdefault(_key(row), []).append(index)
        elif row.get("charged") == "settle" and not row.get("settles"):
            stack = pending.get(_key(row))
            if stack:
                closed.add(stack.pop())
    return [r for i, r in enumerate(rows) if r.get("charged") == "reserved" and i not in closed]


def open_reservations(period: str, path: Optional[Path] = None) -> List[dict]:
    """Return the period's reservations that no settle row has closed.

    A reservation with an `id` is closed by a settle row whose `settles` names it. Rows without an
    `id` pair last in, first out: each settle closes the most recent open reservation of its key.
    """
    return _open(list(entries(path)), period)


MATCH_FIELDS = ("period", "purpose", "staging", "model", "suite", "batch")


def _ref(row: dict) -> str:
    return str(row.get("id") or row.get("ts"))


def settle_open(
    match: Dict[str, str],
    actual: float,
    reason: str,
    path: Optional[Path] = None,
    reservation_id: Optional[str] = None,
) -> dict:
    """Settle the one open reservation `match` names to its real cost `actual` (USD).

    `match` holds period, purpose, staging, model, suite, and batch. With `reservation_id`, settle
    exactly that open reservation, which must also match; without it, exactly one must match.
    Raises LedgerError when none or several do. Returns the appended settle row.
    """

    def decide(items: List[dict]) -> Optional[dict]:
        found = [
            r
            for r in _open(items, match["period"])
            if all(r.get(f) == match[f] for f in MATCH_FIELDS)
        ]
        if reservation_id is not None:
            found = [r for r in found if _ref(r) == reservation_id]
            if not found:
                raise LedgerError(
                    f"reservation {reservation_id} is not open or does not match "
                    f"{json.dumps(match, sort_keys=True)}"
                )
        if len(found) != 1:
            listed = ", ".join(f"{_ref(r)} (ts {r.get('ts')})" for r in found)
            raise LedgerError(
                f"{len(found)} open reservations match {json.dumps(match, sort_keys=True)}; "
                "expected exactly one"
                + (f": {listed}; pass --id <reservation id> to pick one" if found else "")
            )
        held = found[0]
        cap = float(held.get("capUsd", held.get("costUsd", 0.0)))
        row = {f: held[f] for f in MATCH_FIELDS}
        if "claudeVersion" in held:
            row["claudeVersion"] = held["claudeVersion"]
        return {
            **row,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "charged": "settle",
            "costUsd": round(actual - cap, 4),
            "actualUsd": round(actual, 4),
            "reason": reason,
            "settles": held.get("id") or held.get("ts"),
        }

    row = _append_if(path, decide)
    if row is None:
        raise LedgerError("settle_open found no reservation to settle")
    return row


def settle_id(entry: dict, path: Optional[Path] = None) -> bool:
    """Append the settle row `entry` unless a row already settles the same reservation."""

    def decide(items: List[dict]) -> Optional[dict]:
        if any(e.get("settles") == entry["settles"] for e in items if e.get("settles")):
            return None
        return entry

    return _append_if(path, decide) is not None


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
