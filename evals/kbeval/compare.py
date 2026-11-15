"""Compare two baseline summaries and report regressions.

Only cases present in both summaries are compared. A case in one file only is listed as "not run
here", never as removed: a partial rerun after a KB change covers a subset of suites and assets.
"""

from __future__ import annotations

from typing import Dict, List, Optional

DROP = 1 / 3  # one run in three: the smallest drop three runs can show


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _ci(arm: dict) -> str:
    low, high = (arm or {}).get("ci95", [None, None])
    return f"[{_fmt(low)}, {_fmt(high)}]" if low is not None else "[n/a]"


def compare(old: dict, new: dict) -> List[str]:
    """Return report lines for the change from `old` to `new`."""
    lines = [f"{old.get('model')} -> {new.get('model')}, suite {new.get('suite')}"]
    for arm in ("kb", "vanilla", "delta"):
        before = (old.get("aggregates") or {}).get(arm) or {}
        after = (new.get("aggregates") or {}).get(arm) or {}
        lines.append(
            f"suite mean {arm}: {_fmt(before.get('mean'))} {_ci(before)} -> "
            f"{_fmt(after.get('mean'))} {_ci(after)}"
        )
    old_cases: Dict[str, dict] = {c["name"]: c for c in old.get("cases", [])}
    new_cases: Dict[str, dict] = {c["name"]: c for c in new.get("cases", [])}
    regressions = []
    for name in sorted(set(old_cases) & set(new_cases)):
        before, after = old_cases[name], new_cases[name]
        notes = []
        b, a = before.get("kbScore"), after.get("kbScore")
        if b is not None and a is not None and b - a >= DROP - 1e-9:
            notes.append(f"kbScore {b:.2f} -> {a:.2f}")
        if before.get("passed") != after.get("passed"):
            flip = lambda passed: "pass" if passed else "fail"  # noqa: E731
            notes.append(f"{flip(before.get('passed'))} -> {flip(after.get('passed'))}")
        if notes:
            regressions.append(f"  {name}: {'; '.join(notes)}")
    lines.append(f"changed cases ({len(regressions)}):" if regressions else "changed cases: none")
    lines.extend(regressions)
    only = sorted(set(old_cases) ^ set(new_cases))
    if only:
        lines.append(f"not run here ({len(only)}):")
        for name in only:
            side = "old" if name in old_cases else "new"
            lines.append(f"  {name}: not run here (only in {side})")
    return lines
