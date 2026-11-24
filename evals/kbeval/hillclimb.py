"""Trigger-tuning adoption criteria, computed from stored results.

Applies the criteria in `docs/adr/workflow/skill-descriptions-are-tuned-against-held-out-triggers.md`
to trigger rows read through `determination.trigger_rows` (both models together) and effect results
read through `determination.load`. Held-out cases have a `held-` slug, tuning cases `tune-`; an
unprefixed should case is the existing one.

Usage: python3 -m evals.kbeval.hillclimb adopt --asset A [A ...] --before P [P ...]
                                              --after P [P ...] --effects P

Prints `CRITERION <asset> <name> PASS|FAIL|BLOCKED|SKIPPED <counts>` lines and `ADOPT <asset>
adopt|restore|blocked` per asset, then neighbour lines (untuned assets on Sonnet), which change no
verdict. Stderr names each purpose with no results directory (`MISSING purpose ...`) and ends with
`NEIGHBOURS <n> assets checked`. Exits 1 when any asset is `blocked`, or, printing nothing else,
when a (case, model) row is read from more than one purpose (`BLOCKED duplicate ...` on stderr).
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Dict, List, Optional, Tuple

from evals.kbeval import determination as det
from evals.kbeval.results import load_cases, paired_scores, regressed

HELD_GAIN, HELD_RUNS = 4, 12
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _name(value: str) -> str:
    """Return a purpose or asset name that is one plain path component, or raise an argparse error."""
    if not NAME.fullmatch(value):
        raise argparse.ArgumentTypeError(f"not a plain name: {value!r}")
    return value


def _rows(purposes: List[str], assets: List[str]) -> Tuple[List[det.TriggerRow], List[str]]:
    """Return the union of trigger rows over purposes, and duplicate lines.

    A `BLOCKED duplicate` line names each (case, model) found under more than one purpose (a
    purpose named twice, or two purposes holding the same case).
    """
    rows: List[det.TriggerRow] = []
    seen: Dict[Tuple[str, str], List[str]] = {}
    for purpose in purposes:
        for row in det.trigger_rows(purpose, assets):
            rows.append(row)
            seen.setdefault((row[0], row[2]), []).append(purpose)
    dups = [
        f"BLOCKED duplicate {case} {model} in {', '.join(found)}"
        for (case, model), found in seen.items()
        if len(found) > 1
    ]
    return rows, dups


def _split(name: str) -> str:
    """Return a trigger case's split: "held", "tune", or "" (unprefixed)."""
    slug = det._trigger_kind(name)[1]
    return next((p for p in ("held", "tune") if slug.startswith(p + "-")), "")


def _count(rows: List[det.TriggerRow], kind: str, split: Optional[str]) -> Tuple[int, int]:
    """Return (fired, graded) over rows of a kind; split "held", "tune", "" (unprefixed), or None (all)."""
    fired = graded = 0
    for name, row_kind, _model, f, g in rows:
        if row_kind == kind and split in (None, _split(name)):
            fired += f
            graded += g
    return fired, graded


def _held_cases(rows: List[det.TriggerRow]) -> int:
    """Return the number of distinct held-out should cases in rows."""
    return len({name for name, kind, *_ in rows if kind == "trigger-should" and _split(name) == "held"})


def adopt(
    assets: List[str],
    before: List[str],
    after: List[str],
    effects: str,
    effects_base: str = "baseline",
    neighbour_base: str = "baseline",
) -> int:
    """Print the criteria and verdict per asset, then neighbour lines; return 1 on any `blocked`.

    A (case, model) row read from more than one purpose in `before`, `after`, or the neighbour rows
    would count twice: each is printed to stderr as `BLOCKED duplicate ...` and nothing else runs.
    """
    for purpose in dict.fromkeys([*before, *after, effects]):
        path = det.RESULTS / purpose
        if not path.is_dir():
            print(f"MISSING purpose {purpose} ({path})", file=sys.stderr)
    loaded = {}
    unreadable: Dict[str, Exception] = {}
    for asset in assets:
        try:
            loaded[asset] = (_rows(before, [asset]), _rows(after, [asset]))
        except (KeyError, TypeError) as error:
            unreadable[asset] = error
    others = _others(assets, after)
    near_rows, near_dups = _rows(after, others)
    dups = [line for (_, old_dups), (_, new_dups) in loaded.values() for line in old_dups + new_dups]
    if dups or near_dups:
        for line in dups + near_dups:
            print(line, file=sys.stderr)
        return 1
    blocked = False
    for asset in assets:
        if asset in unreadable:
            print(f"BLOCKED unreadable results for {asset}: {unreadable[asset]!r}", file=sys.stderr)
            print(f"ADOPT {asset} blocked")
            blocked = True
            continue
        (old, _), (new, _) = loaded[asset]
        statuses = []

        def emit(name: str, status: str, counts: str) -> None:
            print(f"CRITERION {asset} {name} {status} {counts}")
            statuses.append(status)

        hb, hbn = _count(old, "trigger-should", "held")
        ha, han = _count(new, "trigger-should", "held")
        gain = ha - hb
        why = [
            f"{n} held-out should cases {side}, need 1"
            for side, n in (("before", _held_cases(old)), ("after", _held_cases(new)))
            if n != 1
        ] + [f"{n} graded {side}, need {HELD_RUNS}" for side, n in (("before", hbn), ("after", han)) if n != HELD_RUNS]
        if why:
            status = "BLOCKED"
        else:
            status = "PASS" if gain >= HELD_GAIN else "FAIL"
        emit(
            "held-gain",
            status,
            f"held {hb}/{hbn} -> {ha}/{han} gain {gain:+d} (need >= {HELD_GAIN} of {HELD_RUNS})"
            + "".join(f"; {w}" for w in why),
        )
        eb, ebn = _count(old, "trigger-should", "")
        ea, ean = _count(new, "trigger-should", "")
        why = [f"0 graded {side}" for side, n in (("before", ebn), ("after", ean)) if not n]
        if not why and ebn != ean:
            why = [f"graded {ebn} before, {ean} after"]
        status = "BLOCKED" if why else ("FAIL" if ea < eb else "PASS")
        emit("existing-should", status, f"existing {eb}/{ebn} -> {ea}/{ean}" + "".join(f"; {w}" for w in why))
        nf, nn = _count(new, "trigger-near", None)
        if not nn:
            emit("near-fires", "BLOCKED", f"near {nf}/{nn} fired; 0 graded after")
        else:
            emit("near-fires", "FAIL" if nf else "PASS", f"near {nf}/{nn} fired")
        if all(s == "PASS" for s in statuses):
            try:
                emit("effects", *_effects(asset, effects, effects_base))
            except (KeyError, TypeError) as error:
                print(f"BLOCKED unreadable results for {asset}: {error!r}", file=sys.stderr)
                statuses.append("BLOCKED")
        else:
            emit("effects", "SKIPPED", "a trigger criterion did not pass")
        if "BLOCKED" in statuses:
            verdict = "blocked"
        else:
            verdict = "restore" if "FAIL" in statuses else "adopt"
        blocked = blocked or verdict == "blocked"
        print(f"ADOPT {asset} {verdict}")
    _neighbours(others, near_rows, neighbour_base)
    print(f"NEIGHBOURS {len(others)} assets checked", file=sys.stderr)
    return 1 if blocked else 0


def _effects(asset: str, effects: str, effects_base: str) -> Tuple[str, str]:
    """Return the effects criterion's (status, detail); KeyError or TypeError means a malformed result."""
    found = {m: det.load(effects, m, asset) for m in det.models(asset)}
    data = {m: cases for m, cases in found.items() if cases is not None}
    missing = [m for m in found if m not in data]
    if missing:
        return "BLOCKED", f"no {effects} results for {' '.join(missing)}"
    base = {m: det.load(effects_base, m, asset) for m in data}
    unbased = [m for m, cases in base.items() if cases is None]
    if unbased:
        return "BLOCKED", f"no {effects_base} results for {' '.join(unbased)}"
    pairs = len(paired_scores(base, data, det.SKIP))
    if not pairs:
        return "BLOCKED", f"no effect case compared against {effects_base}"
    bad = regressed(base, data, det.SKIP)
    detail = f"regressed vs {effects_base}: {' '.join(bad)}" if bad else f"no regression vs {effects_base}"
    return ("FAIL" if bad else "PASS"), f"{detail} ({pairs} pairs)"


def _others(assets: List[str], after: List[str]) -> List[str]:
    """Return the assets outside `assets` with Sonnet trigger results under `after`."""
    names = set()
    for purpose in after:
        root = det.RESULTS / purpose / "kb" / det.SONNET / "triggers"
        names |= {n.split("--")[0] for n in load_cases(root).cases if det._trigger_kind(n)[0]}
    return sorted(names - set(assets))


def _neighbours(others: List[str], rows: List[det.TriggerRow], neighbour_base: str) -> None:
    """Print Sonnet near fires and should drops of 2 or more runs for the `others` assets' rows."""
    if not others:
        return
    base = {row[0]: row for row in det.trigger_rows(neighbour_base, others) if row[2] == det.SONNET}
    for name, kind, model, fired, graded in rows:
        if model != det.SONNET:
            continue
        if kind == "trigger-near" and fired:
            print(f"CRITERION neighbours {name} FAIL near {fired}/{graded} fired on {model}")
        elif kind == "trigger-should" and name in base:
            old_fired, old_graded = base[name][3], base[name][4]
            line = f"REVIEW {name} should {old_fired}/{old_graded} -> {fired}/{graded} on {model} vs {neighbour_base}"
            if old_graded != graded:
                print(f"{line} (graded counts differ)")
            elif graded and old_fired - fired >= 2:
                print(line)


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run a subcommand."""
    parser = argparse.ArgumentParser(description="Compute trigger-tuning adoption criteria")
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("adopt")
    a.add_argument("--asset", nargs="+", required=True, type=_name)
    a.add_argument("--before", nargs="+", required=True, type=_name)
    a.add_argument("--after", nargs="+", required=True, type=_name)
    a.add_argument("--effects", required=True, type=_name)
    args = parser.parse_args(argv)
    return adopt(args.asset, args.before, args.after, args.effects)


if __name__ == "__main__":
    sys.exit(main())
