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


class Loaded(NamedTuple):
    """Winning cases, the documents that supply them, their KB digests, and skipped files."""

    cases: Dict[str, dict]
    docs: List[dict]
    digests: Set[str]
    skipped: List[str]


def _tier(stem: str) -> int:
    """Return the tie-break rank of a result file: asset batch 0, exact case 1, `@` rerun 2."""
    if "@" in stem:
        return 2
    return 1 if "--" in stem else 0


def load_cases(root: Path, asset: Optional[str] = None, live: Optional[Set[str]] = None) -> Loaded:
    """Return the winning case per name under one suite directory.

    A file counts only with its `.key` sibling (the runner writes one beside a result it accepted)
    and `partial` false; others go to `skipped`. With `asset` set, only that asset's files are read
    and only its cases kept. `live` drops case names outside it.
    """
    if not root.is_dir():
        return Loaded({}, [], set(), [])
    if asset is None:
        paths = list(root.glob("*.json"))
    else:
        name = glob.escape(asset)
        paths = [root / f"{asset}.json", *root.glob(f"{name}@*.json"), *root.glob(f"{name}--*.json")]
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
    for _, path, doc in found:
        for case in doc.get("cases", []):
            name = case["name"]
            if asset is not None and not name.startswith(asset + "--"):
                continue
            if live is not None and name not in live:
                continue
            winners[name] = (case, path)
    sources = {path for _, path in winners.values()}
    digests = {_digest(path) for path in sources}
    cases = {name: case for name, (case, _) in winners.items()}
    docs = [doc for _, path, doc in found if path in sources]
    return Loaded(cases, docs, digests, sorted(skipped))


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


def evidence(run: dict, name: str) -> Optional[str]:
    """Return the file an llm grader stored for a run, or None."""
    found = grader(run, name)
    return found.get("evidence") if found else None


def own_load_grader(case: dict, asset: str) -> Optional[str]:
    """Return the case's Skill or Agent grader that matches the asset itself, or None."""
    for g in case.get("graders", []):
        cfg = g.get("config") or {}
        if cfg.get("tool") in ("Skill", "Agent") and re.search(
            rf"\b{re.escape(asset)}\"", cfg.get("input_match", "")
        ):
            return g["name"]
    return None
