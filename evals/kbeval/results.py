"""One loader for raw result documents, shared by `summary` and `determination`.

A suite directory `<results>/<purpose>/<staging>/<model>/<suite>/` holds asset batches
(`<asset>.json`), exact-case results (`<asset>--<case>.json`), and API-error reruns
(`<label>@<case>.json`). For one case the newest result by `startedAt` wins; on a tie an
exact-case file beats the asset batch, and an `@` rerun beats both.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

SIDECAR = ".meta.json"
RESULT_TYPES = frozenset({"regex", "file_exists", "llm"})
EPS = 1e-9

Cases = Dict[str, dict]


class Loaded(NamedTuple):
    """Winning cases, their documents and KB digests, skipped files, and live-dropped case names."""

    cases: Dict[str, dict]
    docs: List[dict]
    digests: Set[str]
    skipped: List[str]
    dropped: List[str]


def _tier(stem: str) -> int:
    """Return the tie-break rank of a result file: asset batch 0, exact case 1, `@` rerun 2."""
    if "@" in stem:
        return 2
    return 1 if "--" in stem else 0


def load_cases(root: Path, asset: Optional[str] = None, live: Optional[Set[str]] = None) -> Loaded:
    """Return the winning case per name under one suite directory.

    A file counts only with its `.key` sibling (the runner writes one beside a result it accepted)
    and `partial` false; others go to `skipped`. With `asset` set, only that asset's files are read
    and only its cases kept. `live` drops case names outside it; their names go to `dropped`.
    """
    if not root.is_dir():
        return Loaded({}, [], set(), [], [])
    if asset is None:
        paths = list(root.glob("*.json"))
    else:
        name = glob.escape(asset)
        paths = [
            root / f"{asset}.json",
            *root.glob(f"{name}@*.json"),
            *root.glob(f"{name}--*.json"),
        ]
    found: List[Tuple[Tuple[str, int, str], Path, dict]] = []
    skipped: List[str] = []
    for path in paths:
        if path.name.endswith(SIDECAR) or not path.is_file():
            continue
        if not path.with_suffix(".key").is_file():
            skipped.append(f"{path} (no .key)")
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ValueError(f"{path}: unreadable result ({error})") from error
        if doc.get("partial"):
            skipped.append(f"{path} (partial)")
            continue
        found.append(((doc.get("startedAt") or "", _tier(path.stem), path.stem), path, doc))
    found.sort(key=lambda item: item[0])
    winners: Dict[str, Tuple[dict, Path]] = {}
    dropped: Set[str] = set()
    for _, path, doc in found:
        for case in doc.get("cases", []):
            name = case["name"]
            if asset is not None and not name.startswith(asset + "--"):
                continue
            if live is not None and name not in live:
                dropped.add(name)
                continue
            winners[name] = (case, path)
    sources = {path for _, path in winners.values()}
    digests = {_digest(path) for path in sources}
    cases = {name: case for name, (case, _) in winners.items()}
    docs = [doc for _, path, doc in found if path in sources]
    return Loaded(cases, docs, digests, sorted(skipped), sorted(dropped))


def _digest(path: Path) -> str:
    """Return the `kbDigest` from a result's sidecar, or "unrecorded" when it has none."""
    sidecar = path.with_suffix(SIDECAR)
    if not sidecar.is_file():
        return "unrecorded"
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(f"{sidecar}: unreadable sidecar ({error})") from error
    return meta.get("kbDigest") or "unrecorded"


def runs(case: Optional[dict]) -> List[dict]:
    """Return a case's kb-arm runs that carry grader results."""
    if not case:
        return []
    return [run for run in (case.get("arms") or {}).get("with") or [] if run.get("graders")]


def grader(run: dict, name: str) -> Optional[dict]:
    """Return one grader result from a run, or None."""
    return next((g for g in run["graders"] if g["name"] == name), None)


def passed(run: dict, name: str) -> bool:
    """Return whether the named grader passed in a run."""
    found = grader(run, name)
    return bool(found and found.get("passed"))


def loaded(case: dict, run: dict, name: str) -> bool:
    """Return whether the load grader shows the asset loaded in a run; a `max: 0` grader passes on no load."""
    found = grader(run, name) if run.get("graders") else None
    if found is None:
        return False
    config = next((g.get("config") or {} for g in case.get("graders", []) if g["name"] == name), {})
    return bool(found.get("passed")) != (config.get("max") == 0)


def evidence(run: dict, name: str) -> Optional[str]:
    """Return the file an llm grader stored for a run, or None."""
    found = grader(run, name)
    return found.get("evidence") if found else None


def result_graders(case: dict) -> set:
    """Return the names of a case's result graders."""
    return {g["name"] for g in case.get("graders", []) if g.get("type") in RESULT_TYPES}


def score(case: dict, names: set) -> Optional[float]:
    """Return the mean over runs of the share of `names` graders that passed."""
    scores = []
    for run in runs(case):
        got = [g for g in run["graders"] if g["name"] in names]
        if got:
            scores.append(sum(bool(g.get("passed")) for g in got) / len(got))
    return sum(scores) / len(scores) if scores else None


def paired_scores(
    before: Dict[str, Optional[Cases]], after: Dict[str, Cases], skip: Dict[str, set]
) -> List[Tuple[str, str, float, float]]:
    """Return (case, model, before score, after score) on the result graders both sides share."""
    pairs = []
    for model, cases in after.items():
        for name, case in cases.items():
            old = (before.get(model) or {}).get(name)
            if old is None:
                continue
            kept = result_graders(case) & result_graders(old) - skip.get(name, set())
            if not kept:
                continue
            b, a = score(old, kept), score(case, kept)
            if a is not None and b is not None:
                pairs.append((name, model, b, a))
    return pairs


def regressed(
    before: Dict[str, Optional[Cases]], after: Dict[str, Cases], skip: Dict[str, set]
) -> List[str]:
    """Return the cases whose kb score drops 2/3 on one model, or 1/3 on every model present.

    The 1/3 rule needs at least two models: with one model present, only a 2/3 drop regresses.
    """
    drops: Dict[str, Dict[str, float]] = {}
    for name, model, b, a in paired_scores(before, after, skip):
        drops.setdefault(name, {})[model] = b - a
    return [
        name
        for name, by in drops.items()
        if any(d >= 2 / 3 - EPS for d in by.values())
        or (len(by) >= 2 and all(d >= 1 / 3 - EPS for d in by.values()))
    ]


def own_load_grader(case: dict, asset: str) -> Optional[str]:
    """Return the case's Skill or Agent grader that matches the asset itself, or None."""
    for g in case.get("graders", []):
        cfg = g.get("config") or {}
        if cfg.get("tool") in ("Skill", "Agent") and re.search(
            rf"(?<![\w-]){re.escape(asset)}\"", cfg.get("input_match", "")
        ):
            return g["name"]
    return None
