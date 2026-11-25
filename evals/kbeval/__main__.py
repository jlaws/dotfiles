"""Command line for the KB eval harness: `python3 -m evals.kbeval <command>`."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from evals.kbeval import affected as kaffected
from evals.kbeval import cases as kcases
from evals.kbeval import compare as kcompare
from evals.kbeval import headroom as kheadroom
from evals.kbeval import ledger as kledger
from evals.kbeval import orchestrate, runner
from evals.kbeval import stage as kstage
from evals.kbeval import summary as ksummary

REPO = Path(__file__).resolve().parents[2]


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {number}")
    return number


def _usd_arg(value: str, allow_zero: bool) -> float:
    message = (
        f"expected a number of USD of zero or more, not {value}"
        if allow_zero
        else f"expected a positive number of USD such as 25 or 12.50, not {value}"
    )
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(message) from None
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise argparse.ArgumentTypeError(message)
    return number


def _usd(value: str) -> float:
    return _usd_arg(value, allow_zero=False)


def _nonnegative(value: str) -> float:
    return _usd_arg(value, allow_zero=True)


def _stage(args: argparse.Namespace) -> int:
    try:
        plugin = kstage.stage(REPO, Path(args.out), args.staging, frozenset(args.exclude_tag))
    except kstage.StageError as error:
        print(f"stage: {error}", file=sys.stderr)
        return 1
    print(plugin)
    return 0


def _lint(args: argparse.Namespace) -> int:
    found = kcases.discover(REPO / "evals" / "cases")
    problems = kcases.lint(found, REPO) + kcases.coverage(found, REPO)
    for problem in problems:
        print(problem)
    print(f"{len(found)} cases, {len(problems)} problems")
    return 1 if problems else 0


def _run(args: argparse.Namespace) -> int:
    cap_limit = None
    if args.cap_usd is not None:
        orchestrate.announce_cap(args.purpose, args.cap_usd)
        cap_limit = orchestrate.cap_left(args.purpose, args.cap_usd)
        if cap_limit is None:
            return 2
    status = runner.run(
        args.suite,
        args.staging,
        args.model,
        args.runs,
        args.purpose,
        args.assets,
        args.jobs,
        plugin=Path(args.plugin),
        keep_temp=args.keep_temp,
        cap_limit=cap_limit,
    )
    if args.cap_usd is not None:
        orchestrate.report_cap(args.purpose, args.cap_usd)
    return status


def _pilot(args: argparse.Namespace) -> int:
    return orchestrate.pilot(args.jobs)


def _smoke(args: argparse.Namespace) -> int:
    return orchestrate.smoke(args.assets, args.jobs)


def _baseline(args: argparse.Namespace) -> int:
    return orchestrate.baseline(
        args.model,
        args.vanilla_runs,
        args.vanilla_models,
        frozenset(args.exclude_tag),
        args.jobs,
        cap_usd=args.cap_usd,
    )


def _spend(args: argparse.Namespace) -> int:
    budget = runner.load_budget()
    total = orchestrate.spend()
    print(f"period {budget['period']}: ${total:.2f} of ${float(budget['ceilingUsd']):.2f}")
    return 0 if total <= float(budget["ceilingUsd"]) else 1


def _summarize(args: argparse.Namespace) -> int:
    """Check every summary first and write none unless all pass."""
    settings = {
        "kbRuns": orchestrate.KB_RUNS,
        "vanillaRuns": args.vanilla_runs,
        "vanillaModels": sorted(args.vanilla_models),
        "excludeTags": sorted(args.exclude_tag),
    }
    results = orchestrate.state_root() / "results" / "baseline"
    live = {c.name for c in kcases.discover(REPO / "evals" / "cases")}
    problems: List[str] = []
    notes: List[str] = []
    docs = []
    for model in args.model or list(orchestrate.MODELS):
        for suite in kcases.SUITES:
            try:
                doc = ksummary.summarize(results, model, suite, settings, live=live, notes=notes)
            except ValueError as error:
                problems.append(f"{model}/{suite}: {error}")
                continue
            problems += ksummary.check(doc)
            docs.append(doc)
    for note in notes:
        print(note, file=sys.stderr)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"{len(problems)} problems; no summary written", file=sys.stderr)
        return 1
    for doc in docs:
        out = REPO / "evals" / "baselines" / doc["model"] / f"{doc['suite']}.json"
        ksummary.write(doc, out)
        print(f"wrote {out.relative_to(REPO)} ({len(doc['cases'])} cases)")
    return 0


def _compare(args: argparse.Namespace) -> int:
    try:
        old = json.loads(Path(args.old).read_text(encoding="utf-8"))
        new = json.loads(Path(args.new).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"compare: {error}", file=sys.stderr)
        return 1
    for line in kcompare.compare(old, new):
        print(line)
    return 0


def _settle(args: argparse.Namespace) -> int:
    match = {
        "period": runner.load_budget()["period"],
        "purpose": args.purpose,
        "staging": args.staging,
        "model": args.model,
        "suite": args.suite,
        "batch": args.batch,
    }
    try:
        row = kledger.settle_open(match, args.actual_usd, args.reason)
    except kledger.LedgerError as error:
        print(f"settle: {error}", file=sys.stderr)
        return 1
    print(json.dumps(row, sort_keys=True))
    return 0


def _headroom(args: argparse.Namespace) -> int:
    live: dict = {}
    try:
        for case in kcases.discover(REPO / "evals" / "cases"):
            live.setdefault(case.suite, set()).add(case.name)
        lines = kheadroom.report(REPO / args.baselines, live)
    except (OSError, ValueError) as error:
        print(f"headroom: {error}", file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0


def _git(args: List[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        encoding="utf-8",
        errors="surrogateescape",
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise OSError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def _parse_changes(diff: str, status: str) -> List[Tuple[str, str]]:
    """Return `(status, path)` from NUL-separated `git diff --name-status` and `git status --porcelain` output."""
    tokens = diff.split("\0")
    changes = [(tokens[i][:1], tokens[i + 1]) for i in range(0, len(tokens) - 1, 2)]
    entries = iter(status.split("\0"))
    for entry in entries:
        if entry[3:] and entry[:2] == "??":
            changes.append(("A", entry[3:]))
        elif set(entry[:2]) & {"R", "C"}:
            next(entries, None)
    return changes


def _affected(args: argparse.Namespace) -> int:
    try:
        base = _git(["merge-base", "--end-of-options", args.base, "HEAD"]).strip()
        diff = _git(["diff", "--name-status", "--no-renames", "-z", base])
        status = _git(["status", "--porcelain", "-z", "--untracked-files=all"])
        changes = _parse_changes(diff, status)
        plan = kaffected.affected(REPO, changes)
        for note in plan.notes:
            print(note, file=sys.stderr)
        if plan.everything:
            path = next(p for _, p in changes if p in kaffected.EVERYTHING)
            print(f"ALL <- {path}; a smoke cannot cover it", file=sys.stderr)
            return 2
        if not plan.names:
            print("nothing to run", file=sys.stderr)
        kb, vanilla, cost, unpriced = kaffected.estimate(REPO, plan.names)
        for line in unpriced:
            print(line, file=sys.stderr)
        purpose = f"smoke-{orchestrate._branch().replace('/', '-')}"
        left = orchestrate._capped(purpose, float(runner.load_budget(REPO)["smokeCapUsd"]))
    except (OSError, ValueError, KeyError, kledger.LedgerError) as error:
        print(f"affected: {error}", file=sys.stderr)
        return 1
    print(
        f"ESTIMATE {kb} kb runs + {vanilla} vanilla runs, about ${cost:.2f}; "
        f"smoke cap left ${left:.2f} (purpose {purpose})",
        file=sys.stderr,
    )
    if plan.names:
        print(" ".join(sorted(set(plan.names))))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run one harness command."""
    parser = argparse.ArgumentParser(prog="python3 -m evals.kbeval")
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage", help="stage the kb or vanilla eval plugin")
    stage.add_argument("--out", required=True)
    stage.add_argument("--staging", choices=kstage.STAGINGS, required=True)
    stage.add_argument("--exclude-tag", action="append", default=[])
    stage.set_defaults(func=_stage)
    lint = sub.add_parser("lint", help="lint every case and check coverage")
    lint.set_defaults(func=_lint)

    run = sub.add_parser("run", help="run one suite for one staging and model (paid)")
    run.add_argument("--suite", choices=("triggers", "effects", "workflows"), required=True)
    run.add_argument("--staging", choices=kstage.STAGINGS, required=True)
    run.add_argument("--plugin", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--runs", type=_positive, default=3)
    run.add_argument("--purpose", required=True)
    run.add_argument("--assets", nargs="*", default=[])
    run.add_argument("--jobs", type=_positive, default=8)
    run.add_argument("--keep-temp", action="store_true", help="keep each run's trace")
    run.add_argument(
        "--cap-usd",
        type=_usd,
        help="stop starting runs once the purpose's total spend reaches this; runs in flight can pass it",
    )
    run.set_defaults(func=_run)

    pilot = sub.add_parser("pilot", help="run the pilot cases (paid)")
    pilot.add_argument("--jobs", type=_positive, default=8)
    pilot.set_defaults(func=_pilot)

    smoke = sub.add_parser("smoke", help="one-run smoke for named assets (paid)")
    smoke.add_argument("--assets", nargs="+", required=True)
    smoke.add_argument("--jobs", type=_positive, default=8)
    smoke.set_defaults(func=_smoke)

    baseline = sub.add_parser("baseline", help="full baseline for one model (paid)")
    baseline.add_argument("--model", required=True)
    baseline.add_argument("--vanilla-runs", type=_positive, default=3)
    baseline.add_argument("--vanilla-models", nargs="*", default=list(orchestrate.MODELS))
    baseline.add_argument("--exclude-tag", action="append", default=[])
    baseline.add_argument("--jobs", type=_positive, default=8)
    baseline.add_argument(
        "--cap-usd",
        type=_usd,
        help="stop starting runs once the purpose's total spend reaches this; runs in flight can pass it",
    )
    baseline.set_defaults(func=_baseline)

    summarize = sub.add_parser("summarize", help="write evals/baselines/<model>/<suite>.json")
    summarize.add_argument("--check", action="store_true", help="accepted; summarize always checks")
    summarize.add_argument("--model", action="append")
    summarize.add_argument("--vanilla-runs", type=_positive, default=3)
    summarize.add_argument("--vanilla-models", nargs="*", default=list(orchestrate.MODELS))
    summarize.add_argument("--exclude-tag", action="append", default=[])
    summarize.set_defaults(func=_summarize)

    compare = sub.add_parser("compare", help="regressions between two baseline summaries")
    compare.add_argument("old")
    compare.add_argument("new")
    compare.set_defaults(func=_compare)

    settle = sub.add_parser("settle", help="settle one open reservation to its real cost")
    settle.add_argument("--purpose", required=True)
    settle.add_argument("--staging", choices=kstage.STAGINGS, required=True)
    settle.add_argument("--model", required=True)
    settle.add_argument("--suite", choices=("triggers", "effects", "workflows"), required=True)
    settle.add_argument("--batch", required=True)
    settle.add_argument("--actual-usd", type=_nonnegative, required=True)
    settle.add_argument("--reason", default="interrupted")
    settle.set_defaults(func=_settle)

    headroom = sub.add_parser("headroom", help="list cases that cannot show a change")
    headroom.add_argument("--baselines", default="evals/baselines")
    headroom.set_defaults(func=_headroom)

    affected = sub.add_parser("affected", help="assets a diff touches and the projected smoke cost")
    affected.add_argument("--base", default="origin/main")
    affected.set_defaults(func=_affected)

    spend = sub.add_parser("spend", help="snapshot ledger totals into evals/spend.json")
    spend.set_defaults(func=_spend)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
