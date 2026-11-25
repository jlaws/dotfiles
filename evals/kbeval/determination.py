"""Asset-determination checks: compare kb-arm reruns with the stored baseline, run by run.

Reads raw results under the machine-local state root (`<state>/results/<purpose>/kb/<model>/`), not
the committed summaries, so it runs only where the baseline and reruns were recorded. The checks are
specific to the 2026-11-16 determination (`evals/README.md`, Asset determination).

Usage: python3 -m evals.kbeval.determination <subcommand>

  check --purpose P --asset A [A ...]   per-asset checks; exits 1 on any FAIL
  figures                                determination figures from the baseline results
  spend --purpose P                      purpose spend in the current budget period
  triggers --purpose P --asset A [A ...] [--split tune|held]
                                         per-case trigger fire counts for a purpose
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evals.kbeval import cases as kcases
from evals.kbeval import ledger, runner
from evals.kbeval.orchestrate import state_root
from evals.kbeval.results import (
    EPS,
    Cases,
    evidence,
    grader,
    load_cases,
    loaded,
    own_load_grader,
    paired_scores,
    passed,
    regressed,
    runs,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = state_root() / "results"
OPUS, SONNET = "claude-opus-5-5", "claude-sonnet-5-5"
CONSULT = ("j-arch", "j-cloud", "j-devops", "j-docs", "j-frontend", "j-lang", "j-data")
SKIP = {
    "j-data--effect-orders-schema": {"no-serial"},
    "j-explain--effect-calc-package-explainer": {"self-contained"},
    "j-explain--effect-docs-drift-report": {"self-contained"},
    "architecture-specialist--effect-retry-policy-adr": {"not-retried", "design"},
}
NO_SERIAL_PREFIX = r"^(?:(?!--)[^\n])*?"


def models(asset: str) -> Tuple[str, ...]:
    """Return the models an asset is checked on (writing-plans loads only on Opus)."""
    return (OPUS,) if asset == "writing-plans" else (OPUS, SONNET)


def load(purpose: str, model: str, asset: str, suite: str = "effects") -> Optional[Cases]:
    """Return {case: entry} for one asset, newest result per case; None when no complete result exists."""
    found = load_cases(RESULTS / purpose / "kb" / model / suite, asset=asset)
    return found.cases if found.docs else None


def find_case(cases: Cases, suffix: str) -> Optional[dict]:
    """Return the case whose name ends with `suffix`, or None."""
    return next((c for n, c in cases.items() if n.endswith(suffix)), None)


class Checker:
    """Run the named checks for one purpose and track whether any failed."""

    def __init__(self, purpose: str) -> None:
        """Start with no failures."""
        self.purpose = purpose
        self.failed = False

    def emit(self, name: str, asset: str, status: str, counts: str) -> None:
        """Print one CHECK line and record a FAIL."""
        print(f"CHECK {name} {asset} {status} {counts}")
        self.failed = self.failed or status == "FAIL"

    def batches(self, asset: str) -> Optional[Dict[str, Cases]]:
        """Load every model's results for an asset; emit FAIL and return None if any is missing."""
        out: Dict[str, Cases] = {}
        missing = []
        for model in models(asset):
            cases = load(self.purpose, model, asset)
            if cases is None:
                missing.append(model)
            else:
                out[model] = cases
        if missing:
            self.emit(
                "results", asset, "FAIL", f"missing {self.purpose} results for {' '.join(missing)}"
            )
            return None
        return out

    def ratio(
        self,
        name: str,
        asset: str,
        data: Dict[str, Cases],
        case: str,
        grader_name: str,
        need: float,
    ) -> None:
        """Emit PASS when a grader passes in at least `need` of runs across models."""
        total = good = 0
        parts = []
        for model, cases in data.items():
            rs = runs(find_case(cases, case))
            n = sum(passed(r, grader_name) for r in rs)
            parts.append(f"{model}={n}/{len(rs)}")
            total += len(rs)
            good += n
        ok = total > 0 and good / total >= need - EPS
        self.emit(
            name,
            asset,
            "PASS" if ok else "FAIL",
            f"{grader_name} {good}/{total} ({' '.join(parts)}; need >= {need:.3f})",
        )

    def html_lead(self, data: Dict[str, Cases]) -> None:
        """Check latency-readout `html`, and report runs with the lead before the TOC."""
        self.ratio(
            "html-lead", "html-report-authoring", data, "--effect-latency-readout", "html", 5 / 6
        )
        lead = total = 0
        for cases in data.values():
            for run in runs(find_case(cases, "--effect-latency-readout")):
                text = evidence(run, "html")
                if text is None:
                    continue
                total += 1
                nav = text.find("<nav")
                at = text.find('class="lead"')
                lead += at >= 0 and (nav < 0 or at < nav)
        print(
            f"REPORT html-lead html-report-authoring lead-before-nav {lead}/{total} runs with evidence"
        )

    def writing_plans(self, data: Dict[str, Cases]) -> None:
        """Check that every id-collision run that loaded the skill passes `risks-section`."""
        rs = [
            r
            for r in runs(find_case(data[OPUS], "--effect-id-collision-risk"))
            if passed(r, "skill-loaded")
        ]
        good = sum(passed(r, "risks-section") for r in rs)
        counts = f"risks-section {good}/{len(rs)} loaded runs ({OPUS})"
        status = "BLOCKED" if len(rs) < 2 else ("PASS" if good == len(rs) else "FAIL")
        self.emit("writing-plans-risks", "writing-plans", status, counts)

    def j_explain(self, data: Dict[str, Cases]) -> None:
        """Check `self-contained` across both j-explain cases."""
        total = good = 0
        for cases in data.values():
            for name, case in cases.items():
                for run in runs(case):
                    if grader(run, "self-contained") is not None:
                        total += 1
                        good += passed(run, "self-contained")
        ok = total > 0 and good / total >= 11 / 12 - EPS
        self.emit(
            "j-explain-self-contained",
            "j-explain",
            "PASS" if ok else "FAIL",
            f"self-contained {good}/{total} (need >= 0.917)",
        )

    def no_regression(self, asset: str, data: Dict[str, Cases]) -> None:
        """Fail a case whose kb score drops 1/3 on both models or 2/3 on one (`results.regressed`)."""
        base = {m: load("baseline", m, asset) for m in data}
        lines = [
            f"{name}@{model.split('-')[1]}={b:.2f}->{a:.2f}"
            for name, model, b, a in paired_scores(base, data, SKIP)
        ]
        bad = regressed(base, data, SKIP)
        status = "FAIL" if bad else "PASS"
        detail = " ".join(lines) + (f" regressed: {' '.join(bad)}" if bad else "")
        self.emit("no-regression", asset, status, detail)

    def library_use(self, asset: str, data: Dict[str, Cases]) -> None:
        """Report runs that delegated and runs that read references inline."""
        parts = []
        for model, cases in data.items():
            rs = [r for c in cases.values() for r in runs(c)]
            delegated = sum(passed(r, "delegated") for r in rs)
            inline = sum(passed(r, "read-refs") and not passed(r, "delegated") for r in rs)
            parts.append(
                f"{model}: delegated {delegated}/{len(rs)} inline-read-refs {inline}/{len(rs)}"
            )
        print(f"REPORT library-use {asset} {'; '.join(parts)}")


def check(purpose: str, assets: List[str]) -> int:
    """Run every check for the named assets; return 1 on any FAIL."""
    checker = Checker(purpose)
    for asset in assets:
        data = checker.batches(asset)
        if data is None:
            continue
        if asset == "html-report-authoring":
            checker.html_lead(data)
        elif asset == "writing-plans":
            checker.writing_plans(data)
        elif asset == "research-analyst":
            checker.ratio(
                "research-claim-c", asset, data, "--effect-claim-confidence", "claims", 11 / 12
            )
        elif asset == "j-explain":
            checker.j_explain(data)
        elif asset == "architecture-specialist":
            checker.ratio("arch-design", asset, data, "--effect-retry-policy-adr", "design", 5 / 6)
        checker.no_regression(asset, data)
        if asset in CONSULT or asset == "j-system-design":
            checker.library_use(asset, data)
    return 1 if checker.failed else 0


def skill_assets() -> List[str]:
    """Return every Claude skill and command name."""
    skills = sorted(
        p.name for p in (REPO / ".claude" / "skills").iterdir() if (p / "SKILL.md").is_file()
    )
    commands = sorted(p.stem for p in (REPO / ".claude" / "commands").glob("*.md"))
    return skills + commands


def rescore(data: Cases, case_suffix: str, regex_name: str, prefix: str, not_contains: bool) -> str:
    """Re-score a fixed regex grader on files stored by an llm grader in the same case."""
    case = find_case(data, case_suffix)
    if case is None:
        return "no case"
    cfgs = {g["name"]: g for g in case["graders"]}
    cfg = cfgs[regex_name]["config"]
    path = cfg["target"]["path"]
    llm = next(
        (
            n
            for n, g in cfgs.items()
            if g["type"] == "llm"
            and ((g.get("config") or {}).get("focus") or {}).get("path") == path
        ),
        None,
    )
    if llm is None:
        return "no llm grader stores the file"
    pattern = re.compile(prefix + cfg["pattern"], kcases._python_flags(cfg.get("flags", "")))
    stored = new = total = 0
    for run in runs(case):
        text = evidence(run, llm)
        if text is None:
            continue
        total += 1
        stored += passed(run, regex_name)
        new += (pattern.search(text) is not None) != not_contains
    return f"stored-pass {stored}/{total} fixed-pass {new}/{total} (evidence via {llm})"


def figures() -> int:
    """Print the determination figures from the baseline results."""
    print("(a) never-firing assets: own Skill grader passes over trigger-should + effect kb runs")
    for asset in skill_assets():
        good = total = 0
        for model in (OPUS, SONNET):
            for suite, marker in (("triggers", "--trigger-should-"), ("effects", "--effect-")):
                for name, case in (load("baseline", model, asset, suite) or {}).items():
                    if marker not in name:
                        continue
                    g = own_load_grader(case, asset)
                    if g is None:
                        continue
                    for run in runs(case):
                        total += 1
                        good += passed(run, g)
        if total and good == 0:
            print(f"  {asset}: 0/{total}")
    print("(b) delegated passes over kb runs")
    for asset in (*CONSULT, "j-system-design"):
        parts = []
        for model in (OPUS, SONNET):
            rs = [r for c in (load("baseline", model, asset) or {}).values() for r in runs(c)]
            parts.append(f"{model}={sum(passed(r, 'delegated') for r in rs)}/{len(rs)}")
        print(f"  {asset}: {' '.join(parts)}")
    print("(c) kb runs whose stored evidence cites .kb-references/ (lower bound on library reads)")
    for asset in (*CONSULT, "j-system-design"):
        cited = total = 0
        for model in (OPUS, SONNET):
            for case in (load("baseline", model, asset) or {}).values():
                for run in runs(case):
                    texts = [g.get("evidence") for g in run["graders"] if g.get("evidence")]
                    if texts:
                        total += 1
                        cited += any(".kb-references/" in t for t in texts)
        print(f"  {asset}: {cited}/{total} runs with evidence")
    print("(d) fixed graders re-scored on stored evidence")
    for model in (OPUS, SONNET):
        ex = load("baseline", model, "j-explain") or {}
        for suffix in ("--effect-calc-package-explainer", "--effect-docs-drift-report"):
            print(
                f"  j-explain{suffix}/self-contained {model}: {rescore(ex, suffix, 'self-contained', '', True)}"
            )
        data = load("baseline", model, "j-data") or {}
        res = rescore(data, "--effect-orders-schema", "no-serial", NO_SERIAL_PREFIX, True)
        print(f"  j-data--effect-orders-schema/no-serial {model}: {res}")
    return 0


def spend(purpose: str) -> int:
    """Print the purpose's spend in the current budget period."""
    period = runner.load_budget(REPO)["period"]
    print(f"{ledger.spent(period, purpose=purpose):.2f} USD (period {period}, purpose {purpose})")
    return 0


TRIGGER_KINDS = ("trigger-should", "trigger-near", "trigger-noauto")
TriggerRow = Tuple[str, str, str, int, int]


def _trigger_kind(name: str) -> Tuple[str, str]:
    """Split a trigger case name into (kind, slug); kind is "" for a non-trigger case."""
    rest = name.split("--", 1)[1] if "--" in name else ""
    for kind in TRIGGER_KINDS:
        if rest.startswith(kind + "-"):
            return kind, rest[len(kind) + 1 :]
    return "", ""


def trigger_rows(purpose: str, assets: List[str], split: Optional[str] = None) -> List[TriggerRow]:
    """Return (case, kind, model, fired runs, graded runs) for each trigger case under a purpose.

    Reads `<asset>.json`, `<asset>@*.json`, and exact-case `<asset>--*.json` files under
    `results/<purpose>/kb/<model>/triggers/` through `results.load_cases` (newest `startedAt` wins
    for a case); skipped files are reported on stderr. A run counts as graded only when the case's
    own load grader (`results.own_load_grader`) is present in it; fired counts graded runs that
    `results.loaded` says loaded the skill (a `max: 0` grader, as on trigger-near and
    trigger-noauto cases, passes when the skill was not called). A case with no own load grader
    is reported on stderr as `NO LOAD GRADER <case>` and keeps 0 graded runs. `split` "held"
    keeps cases whose slug starts "held-"; "tune" keeps every other case, unprefixed ones included.
    """
    rows: List[TriggerRow] = []
    base = RESULTS / purpose / "kb"
    model_dirs = sorted(d for d in base.iterdir() if d.is_dir()) if base.is_dir() else []
    for model_dir in model_dirs:
        for asset in assets:
            found = load_cases(model_dir / "triggers", asset=asset)
            for skipped in found.skipped:
                print(f"SKIPPED {skipped}", file=sys.stderr)
            for name, case in found.cases.items():
                kind, slug = _trigger_kind(name)
                held = slug.startswith("held-")
                if not kind or (split == "held" and not held) or (split == "tune" and held):
                    continue
                load_grader = own_load_grader(case, asset)
                if load_grader is None:
                    print(f"NO LOAD GRADER {name}", file=sys.stderr)
                    rows.append((name, kind, model_dir.name, 0, 0))
                    continue
                graded = [r for r in runs(case) if grader(r, load_grader)]
                fired = sum(loaded(case, r, load_grader) for r in graded)
                rows.append((name, kind, model_dir.name, fired, len(graded)))
    return sorted(rows, key=lambda row: (row[0], row[2]))


def triggers(purpose: str, assets: List[str], split: Optional[str]) -> int:
    """Print one TRIGGER line per trigger case and model; exit 1 when nothing matched."""
    rows = trigger_rows(purpose, assets, split)
    for name, kind, model, fired, graded in rows:
        print(f"TRIGGER {name} {kind} {model} {fired}/{graded}")
    if not rows:
        print(
            f"no trigger results for purpose={purpose} assets={' '.join(assets)}", file=sys.stderr
        )
        return 1
    return 0


def main() -> int:
    """Parse arguments and run a subcommand."""
    parser = argparse.ArgumentParser(
        description="Compare kb-arm eval reruns with the stored baseline"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--purpose", required=True)
    c.add_argument("--asset", nargs="+", required=True)
    sub.add_parser("figures")
    s = sub.add_parser("spend")
    s.add_argument("--purpose", required=True)
    t = sub.add_parser("triggers")
    t.add_argument("--purpose", required=True)
    t.add_argument("--asset", nargs="+", required=True)
    t.add_argument("--split", choices=("tune", "held"))
    args = parser.parse_args()
    if args.cmd == "check":
        return check(args.purpose, args.asset)
    if args.cmd == "figures":
        return figures()
    if args.cmd == "triggers":
        return triggers(args.purpose, args.asset, args.split)
    return spend(args.purpose)


if __name__ == "__main__":
    sys.exit(main())
