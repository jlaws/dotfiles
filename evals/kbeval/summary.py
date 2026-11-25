"""Compact per-suite baseline summaries from raw result documents.

Reads `<results>/<staging>/<model>/<suite>/` through `results.load_cases` (newest result per case
wins), and scores each case on its result graders only (`regex`, `file_exists`,
`llm`). Process graders (`tool_used`, `tool_order`) are left out of delta: the asset's own load
grader gives `loadRate`, the others give `behaviorRate`. Trigger cases have only a process grader, so it is their score.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

from evals.kbeval.cases import CASE_NAME
from evals.kbeval.results import load_cases, loaded, own_load_grader
from evals.kbeval.runner import API_ERROR, JUDGE_MODEL, TIMEOUT, TURN_CAP, judge_family

SCHEMA = 1
RESULT_TYPES = frozenset({"regex", "file_exists", "llm"})
BOOTSTRAP_SAMPLES = 2000
PASS_AT = 2 / 3  # a case passes when its mean run score is at least 2/3
# Summaries are committed to a public repo: refuse keys, local paths, and API request IDs.
LEAK = re.compile(
    r"sk-ant-|\bsk-[0-9A-Za-z_-]{20,}|\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[0-9A-Za-z]{20,}"
    r"|/Users/|/home/|/var/folders/|/tmp/|/private/|(?<![\w.])~/"
    r"|\b(?:req|msg)_[0-9A-Za-z]{6,}"
)


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


def _behavior_rate(case: dict, load: Optional[str]) -> Optional[float]:
    """Return the pass rate of the process graders other than the load grader, or None."""
    types = {g["name"]: g.get("type") for g in case.get("graders", [])}
    results = [
        bool(g.get("passed"))
        for run in _graded(case)
        for g in run["graders"]
        if types.get(g["name"]) not in RESULT_TYPES and g["name"] != load
    ]
    return sum(results) / len(results) if results else None


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


def summarize(
    results: Path, model: str, suite: str, settings: dict, live: Optional[Set[str]] = None,
    notes: Optional[List[str]] = None,
) -> dict:
    """Return the summary document for one model and suite.

    `settings` records the run configuration (`kbRuns`, `vanillaRuns`, `vanillaModels`,
    `excludeTags`); `check` holds the summary to it. `live` drops results for cases outside it.
    `notes`, when given, receives one line per skipped file and one per staging with dropped cases.
    """
    by_staging = {staging: load_cases(results / staging / model / suite, live=live) for staging in ("kb", "vanilla")}
    if notes is not None:
        for staging, found in by_staging.items():
            where = f"{staging}/{model}/{suite}"
            notes.extend(f"{where}: skipped {entry}" for entry in found.skipped)
            if found.dropped:
                notes.append(
                    f"{where}: dropped results for {len(found.dropped)} cases no longer under "
                    f"evals/cases: {', '.join(found.dropped)}"
                )
    kb_cases, kb_docs, kb_digests, *_ = by_staging["kb"]
    van_cases, van_docs, van_digests, *_ = by_staging["vanilla"]
    for digests in (kb_digests, van_digests):
        if len(digests) > 1:
            raise ValueError(
                f"results mix KB digests: {', '.join(sorted(digests))}; "
                "rerun that staging or move the older results aside"
            )
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
        load = own_load_grader(kb, _identity(name)[0]) if kb else None
        behavior = _behavior_rate(kb, load) if kb else None
        capped = [
            (is_kb, r)
            for is_kb, c in ((True, kb), (False, van))
            if c
            for r in _runs(c)
            if (error := r.get("error")) and (TURN_CAP.search(error) or TIMEOUT.search(error))
        ]
        load_rate, censored = None, 0
        if kb and load:
            kb_graded = _graded(kb)
            if kb_graded:
                load_rate = sum(loaded(kb, r, load) for r in kb_graded) / len(kb_graded)
            censored = sum(1 for is_kb, r in capped if is_kb and not loaded(kb, r, load))
        errors = sorted({str(r["error"]) for c in (kb, van) if c for r in _runs(c) if r.get("error")})
        error_runs = sum(1 for c in (kb, van) if c for r in _runs(c) if r.get("error"))
        asset, kind = _identity(name)
        rows.append(
            {
                "asset": asset,
                "behaviorRate": None if behavior is None else round(behavior, 4),
                "cappedRuns": len(capped),
                "censoredRuns": censored,
                "delta": None if kb_score is None or van_score is None else round(kb_score - van_score, 4),
                "errorRuns": error_runs,
                "errors": errors,
                "kbRuns": kb_runs,
                "kbScore": None if kb_score is None else round(kb_score, 4),
                "kind": kind,
                "loadRate": None if load_rate is None else round(load_rate, 4),
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
        "kbDigest": {
            "kb": next(iter(kb_digests), None),
            "vanilla": next(iter(van_digests), None),
        },
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
    # A summary with no recorded judge (triggers) is held to the runner's judge, as `runner.run` is.
    if judge_family(doc["model"]) == judge_family(doc.get("judgeModel") or JUDGE_MODEL):
        problems.append(f"{doc['model']}/{doc['suite']}: judged by its own family")
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
    leak = LEAK.search(text)
    if leak:
        raise ValueError(f"refusing to write {out}: the summary contains {leak.group(0)!r}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
