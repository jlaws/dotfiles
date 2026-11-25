"""Compare two baseline summaries and report regressions.

A case is named only when kbScore fell by 2/3 or more; a smaller drop or a pass/fail flip is counted.
A `passed` value of None on either side is unknown, not a flip.
Summaries judged by different models raise `IncomparableError`.

Only cases present in both summaries are compared. A case in one file only is listed as "not run
here", never as removed: a partial rerun after a KB change covers a subset of suites and assets.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from evals.kbeval.results import NOISE_DROP, REGRESSED_DROP

TOLERANCE = 1e-3  # summaries round scores to 4 decimals


class IncomparableError(ValueError):
    """Two summaries were judged by different models, so their scores do not compare."""


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _flip(passed: Optional[bool]) -> str:
    return "pass" if passed else "fail"


def _ci(arm: dict) -> str:
    low, high = (arm or {}).get("ci95") or [None, None]
    return f"[{_fmt(low)}, {_fmt(high)}]" if low is not None else "[n/a]"


def _version_line(old: dict, new: dict) -> str:
    judge = new.get("judgeModel") or "unknown"
    before = old.get("claudeVersion") or "unknown"
    after = new.get("claudeVersion") or "unknown"
    if before == after:
        return f"judge {judge}; Claude Code {after}"
    return (
        f"judge {judge}; Claude Code {before} -> {after} "
        "(differs: a change below may come from the CLI, not the KB)"
    )


def compare(old: dict, new: dict) -> List[str]:
    """Return report lines for the change from `old` to `new`."""
    judges = (old.get("judgeModel"), new.get("judgeModel"))
    if all(judges) and judges[0] != judges[1]:
        raise IncomparableError(f"judge models differ ({judges[0]} vs {judges[1]})")
    lines = [
        f"{old.get('model')} -> {new.get('model')}, suite {new.get('suite')}",
        _version_line(old, new),
    ]
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
    noise = 0
    for name in sorted(set(old_cases) & set(new_cases)):
        before, after = old_cases[name], new_cases[name]
        notes = []
        b, a = before.get("kbScore"), after.get("kbScore")
        drop = b is not None and a is not None and b - a >= REGRESSED_DROP - TOLERANCE
        known = before.get("passed") is not None and after.get("passed") is not None
        flipped = known and before.get("passed") != after.get("passed")
        if drop:
            notes.append(f"kbScore {b:.2f} -> {a:.2f}")
        if drop and flipped:
            notes.append(f"{_flip(before.get('passed'))} -> {_flip(after.get('passed'))}")
        if notes:
            regressions.append(f"  {name}: {'; '.join(notes)}")
        elif flipped or (b is not None and a is not None and b - a >= NOISE_DROP - TOLERANCE):
            noise += 1
    lines.append(
        f"regressed ({len(regressions)}): kbScore dropped by 2/3 or more"
        if regressions
        else "regressed: none"
    )
    lines.extend(regressions)
    lines.append(
        f"within run-to-run noise: {noise} cases "
        "(a drop of 1/3 up to 2/3, or a pass/fail flip; not listed)"
    )
    only = sorted(set(old_cases) ^ set(new_cases))
    if only:
        lines.append(f"not run here ({len(only)}):")
        for name in only:
            side = "old" if name in old_cases else "new"
            lines.append(f"  {name}: not run here (only in {side})")
    return lines
