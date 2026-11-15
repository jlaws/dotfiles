"""Compact per-suite baseline summaries from raw result documents.

Reads `<results>/<staging>/<model>/<suite>/<batch>.json` (and `<batch>@<case>.json` reruns, which
replace that case), and scores each case on its result graders only (`regex`, `file_exists`,
`llm`). Process graders (`tool_used`, `tool_order`) are indicators, reported as `firedRate` and
left out of delta. Trigger cases have only a process grader, so it is their score.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evals.kbeval.cases import CASE_NAME
from evals.kbeval.runner import API_ERROR

SCHEMA = 1
RESULT_TYPES = frozenset({"regex", "file_exists", "llm"})
BOOTSTRAP_SAMPLES = 2000
PASS_AT = 2 / 3  # a case passes when its mean run score is at least 2/3


def _load(results: Path, staging: str, model: str, suite: str) -> Tuple[Dict[str, dict], List[dict]]:
    """Return {case name: case entry} with reruns applied, and every document read."""
    root = results / staging / model / suite
    cases: Dict[str, dict] = {}
    docs: List[dict] = []
    if not root.is_dir():
        return cases, docs
    # Only completed batches count: the runner writes a .key beside a result it accepted, so
    # partial, quarantined, or abandoned results are left out.
    paths = sorted(p for p in root.glob("*.json") if p.with_suffix(".key").is_file())
    for path in [p for p in paths if "@" not in p.stem] + [p for p in paths if "@" in p.stem]:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("partial"):
            continue
        docs.append(doc)
        for case in doc.get("cases", []):
            cases[case["name"]] = case
    return cases, docs


def _runs(case: dict) -> List[dict]:
    return [run for arm in (case.get("arms") or {}).values() for run in arm or []]


def _graded(case: dict) -> List[dict]:
    """Return runs that carry grader results (an API-error run has none)."""
    return [run for run in _runs(case) if run.get("graders")]


def _score(case: dict, run: dict, result: bool) -> Optional[float]:
    types = {g["name"]: g.get("type") for g in case.get("graders", [])}
    weights = {g["name"]: float(g.get("weight", 1)) for g in case.get("graders", [])}
    chosen = [g for g in run.get("graders", []) if (types.get(g["name"]) in RESULT_TYPES) == result]
    total = sum(weights.get(g["name"], 1.0) for g in chosen)
    if not chosen or total == 0:
        return None
    return sum(weights.get(g["name"], 1.0) for g in chosen if g.get("passed")) / total


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _arm_score(case: Optional[dict], result: bool) -> Tuple[Optional[float], int]:
    if case is None:
        return None, 0
    runs = _graded(case)
    scores = [s for s in (_score(case, run, result) for run in runs) if s is not None]
    return _mean(scores), len(runs)


def _bootstrap(values: List[float]) -> dict:
    if not values:
        return {"mean": None, "ci95": [None, None]}
    rng = random.Random(0)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(BOOTSTRAP_SAMPLES)
    )
    low = means[int(0.025 * (BOOTSTRAP_SAMPLES - 1))]
    high = means[int(0.975 * (BOOTSTRAP_SAMPLES - 1))]
    return {"mean": round(sum(values) / len(values), 4), "ci95": [round(low, 4), round(high, 4)]}


def _identity(name: str) -> Tuple[str, str]:
    match = CASE_NAME.match(name)
    if match:
        return match.group("asset"), match.group("kind")
    return name, "workflow"


def summarize(results: Path, model: str, suite: str, settings: dict) -> dict:
    """Return the summary document for one model and suite.

    `settings` records the run configuration (`kbRuns`, `vanillaRuns`, `vanillaModels`,
    `excludeTags`); `check` holds the summary to it.
    """
    kb_cases, kb_docs = _load(results, "kb", model, suite)
    van_cases, van_docs = _load(results, "vanilla", model, suite)
    docs = kb_docs + van_docs
    versions = sorted({d.get("claudeVersion") for d in docs if d.get("claudeVersion")})
    if len(versions) > 1:
        raise ValueError(f"results mix Claude Code versions: {', '.join(versions)}")
    judges = sorted({(d.get("suite") or {}).get("judgeModel") for d in docs} - {None})
    if len(judges) > 1:
        raise ValueError(f"results mix judge models: {', '.join(judges)}")
    trigger = suite == "triggers"
    rows = []
    for name in sorted(set(kb_cases) | set(van_cases)):
        kb, van = kb_cases.get(name), van_cases.get(name)
        kb_score, kb_runs = _arm_score(kb, result=not trigger)
        van_score, van_runs = _arm_score(van, result=not trigger)
        fired, _ = _arm_score(kb, result=False)
        errors = sorted({str(r["error"]) for c in (kb, van) if c for r in _runs(c) if r.get("error")})
        asset, kind = _identity(name)
        rows.append(
            {
                "asset": asset,
                "delta": None if kb_score is None or van_score is None else round(kb_score - van_score, 4),
                "errors": errors,
                "firedRate": None if fired is None else round(fired, 4),
                "kbRuns": kb_runs,
                "kbScore": None if kb_score is None else round(kb_score, 4),
                "kind": kind,
                "name": name,
                "passed": kb_score is not None and kb_score >= PASS_AT - 1e-9,
                "vanillaRuns": van_runs,
                "vanillaScore": None if van_score is None else round(van_score, 4),
            }
        )
    dates = sorted(d.get("startedAt", "")[:10] for d in docs if d.get("startedAt"))
    return {
        "schema": SCHEMA,
        "suite": suite,
        "model": model,
        "judgeModel": judges[0] if judges else None,
        "claudeVersion": versions[0] if versions else None,
        "date": dates[-1] if dates else None,
        "runs": settings,
        "costUsd": round(sum(float(d.get("costUsd") or 0.0) for d in docs), 4),
        "aggregates": {
            "kb": _bootstrap([r["kbScore"] for r in rows if r["kbScore"] is not None]),
            "vanilla": _bootstrap([r["vanillaScore"] for r in rows if r["vanillaScore"] is not None]),
            "delta": _bootstrap([r["delta"] for r in rows if r["delta"] is not None]),
        },
        "cases": rows,
    }


def check(doc: dict) -> List[str]:
    """Return problems that make a summary unfit to commit."""
    settings = doc.get("runs") or {}
    want_vanilla = doc["suite"] != "triggers" and doc["model"] in settings.get("vanillaModels", [])
    problems = []
    if not doc.get("cases"):
        problems.append(f"{doc['model']}/{doc['suite']}: no cases")
    for case in doc.get("cases", []):
        name = case["name"]
        if case["kbRuns"] != settings.get("kbRuns"):
            problems.append(f"{name}: kbRuns {case['kbRuns']}, expected {settings.get('kbRuns')}")
        if want_vanilla and case["vanillaRuns"] != settings.get("vanillaRuns"):
            problems.append(f"{name}: vanillaRuns {case['vanillaRuns']}, expected {settings.get('vanillaRuns')}")
        for error in case.get("errors", []):
            if API_ERROR.search(error):
                problems.append(f"{name}: unresolved API error: {error}")
    return problems


def write(doc: dict, out: Path) -> None:
    """Write a summary with sorted keys and one case per line."""
    head = {k: v for k, v in doc.items() if k != "cases"}
    lines = ["{"]
    for key in sorted(head):
        lines.append(f"  {json.dumps(key)}: {json.dumps(head[key], sort_keys=True)},")
    lines.append('  "cases": [')
    rows = [json.dumps(case, sort_keys=True) for case in doc["cases"]]
    lines.extend(f"    {row}," for row in rows[:-1])
    if rows:
        lines.append(f"    {rows[-1]}")
    lines.append("  ]")
    lines.append("}")
    text = "\n".join(lines) + "\n"
    if "sk-ant-" in text:
        raise ValueError(f"refusing to write {out}: the summary contains an sk-ant- string")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
