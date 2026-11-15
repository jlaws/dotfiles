"""The paid run plans behind the make targets: pilot, smoke, and baseline.

Each plan stages fresh plugins under the machine-local state directory, then calls `runner.run`
per suite and staging, stopping at the first non-zero status.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

from evals.kbeval import ledger, runner
from evals.kbeval.cases import assets as repo_assets
from evals.kbeval.stage import stage

REPO = runner.REPO
MODELS = ("claude-opus-5-5", "claude-sonnet-5-5")
SMOKE_MODEL = "claude-sonnet-5-5"
PILOT_ASSETS = ("design-first", "j-tdd", "j-arch", "j-plan", "test-driven-development")
PILOT_CAP_USD = 10.0
KB_RUNS = 3
VANILLA_SAMPLE = 5


def state_root() -> Path:
    """Return the machine-local state directory beside the ledger."""
    return ledger.default_path().parent


def _stage(staging: str, exclude_tags: frozenset = frozenset()) -> Path:
    suffix = "-x-" + "-".join(sorted(exclude_tags)) if exclude_tags else ""
    return stage(REPO, state_root() / "stage" / f"{staging}{suffix}", staging, exclude_tags)


def _capped(purpose: str, cap_usd: float) -> float:
    budget = runner.load_budget(REPO)
    return cap_usd - ledger.spent(budget["period"], purpose=purpose)


def _sequence(steps: Iterable[tuple], cli: Optional[runner.Cli]) -> int:
    for suite, staging, model, runs, purpose, assets, jobs, options in steps:
        status = runner.run(suite, staging, model, runs, purpose, assets, jobs, cli, **options)
        if status != 0:
            return status
    return 0


def _effect_cases(plugin: Path, assets: List[str]) -> List[str]:
    root = plugin / "evals-effects"
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir() if d.is_dir() and any(d.name.startswith(f"{a}--") for a in assets)
    )


def pilot(jobs: int = 8, cli: Optional[runner.Cli] = None) -> int:
    """Run the pilot cases once on both models: triggers on kb, effects on kb and vanilla."""
    kb, vanilla = _stage("kb"), _stage("vanilla")
    assets = list(PILOT_ASSETS)
    for model in MODELS:
        for suite, staging, plugin in (("triggers", "kb", kb), ("effects", "kb", kb), ("effects", "vanilla", vanilla)):
            options = {"plugin": plugin, "state": state_root() / "results",
                       "cap_limit": _capped("pilot", PILOT_CAP_USD)}  # fmt: skip
            status = _sequence([(suite, staging, model, 1, "pilot", assets, jobs, options)], cli)
            if status != 0:
                return status
    return 0


def _branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, capture_output=True, text=True,
        stdin=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() or "unknown"


def smoke(assets: List[str], jobs: int = 8, cli: Optional[runner.Cli] = None) -> int:
    """Run kb once per case on Sonnet for `assets`, then vanilla on their first effect cases."""
    budget = runner.load_budget(REPO)
    purpose = f"smoke-{_branch().replace('/', '-')}"
    kb, vanilla = _stage("kb"), _stage("vanilla")
    workflow_cases = [a for a in assets if a.startswith("wf--")]
    plain = [a for a in assets if not a.startswith("wf--")]
    steps = [("triggers", plain), ("effects", plain), ("workflows", workflow_cases)]
    for suite, chosen in steps:
        if not chosen:
            continue
        options = {"plugin": kb, "state": state_root() / "results",
                   "cap_limit": _capped(purpose, float(budget["smokeCapUsd"])), "keep_temp": True}  # fmt: skip
        status = runner.run(suite, "kb", SMOKE_MODEL, 1, purpose, chosen, jobs, cli, **options)
        if status != 0:
            return status
    for name in _effect_cases(vanilla, plain)[:VANILLA_SAMPLE]:
        options = {"plugin": vanilla, "state": state_root() / "results",
                   "cap_limit": _capped(purpose, float(budget["smokeCapUsd"])), "keep_temp": True}  # fmt: skip
        status = runner.run("effects", "vanilla", SMOKE_MODEL, 1, purpose, [name], jobs, cli, **options)
        if status != 0:
            return status
    return 0


def baseline(
    model: str,
    vanilla_runs: int = 3,
    vanilla_models: Iterable[str] = MODELS,
    exclude_tags: frozenset = frozenset(),
    jobs: int = 8,
    cli: Optional[runner.Cli] = None,
) -> int:
    """Run the full baseline for one model: kb at three runs per case, vanilla as configured."""
    kb = _stage("kb", exclude_tags)
    assets = sorted(repo_assets(REPO))
    state = state_root() / "results"
    steps = [
        ("triggers", "kb", model, KB_RUNS, "baseline", assets, jobs, {"plugin": kb, "state": state}),
        ("effects", "kb", model, KB_RUNS, "baseline", assets, jobs, {"plugin": kb, "state": state}),
        ("workflows", "kb", model, KB_RUNS, "baseline", [], jobs, {"plugin": kb, "state": state}),
    ]
    if model in set(vanilla_models):
        vanilla = _stage("vanilla", exclude_tags)
        steps += [
            ("effects", "vanilla", model, vanilla_runs, "baseline", assets, jobs, {"plugin": vanilla, "state": state}),
            ("workflows", "vanilla", model, vanilla_runs, "baseline", [], jobs, {"plugin": vanilla, "state": state}),
        ]  # fmt: skip
    return _sequence(steps, cli)


def spend(out: Optional[Path] = None) -> float:
    """Snapshot the current period's ledger totals into evals/spend.json and return the total."""
    budget = runner.load_budget(REPO)
    target = out or REPO / "evals" / "spend.json"
    ledger.snapshot(target, budget["period"])
    return ledger.spent(budget["period"])
