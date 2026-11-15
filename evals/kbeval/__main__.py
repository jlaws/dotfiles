"""Command line for the KB eval harness: `python3 -m evals.kbeval <command>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from evals.kbeval import cases as kcases
from evals.kbeval import stage as kstage

REPO = Path(__file__).resolve().parents[2]


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
    problems = kcases.lint(found, REPO) + kcases.coverage(found, REPO, kcases.PENDING_EVAL_COVERAGE)
    for problem in problems:
        print(problem)
    print(f"{len(found)} cases, {len(problems)} problems")
    return 1 if problems else 0


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
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
