"""Asset-determination checks: compare kb-arm reruns with the stored baseline, run by run.

Reads raw results under the machine-local state root (`<state>/results/<purpose>/kb/<model>/`), not
the committed summaries, so it runs only where the baseline and reruns were recorded. The checks are
specific to the 2026-11-16 determination (`evals/README.md`, Asset determination).

Usage: python3 -m evals.kbeval.determination <subcommand>

  check --purpose P --asset A [A ...]   per-asset checks; exits 1 on any FAIL
  figures                                determination figures from the baseline results
  spend --purpose P                      purpose spend in the current budget period
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evals.kbeval import cases as kcases
from evals.kbeval import ledger, runner
from evals.kbeval.orchestrate import state_root

REPO = Path(__file__).resolve().parents[2]
RESULTS = state_root() / "results"
OPUS, SONNET = "claude-opus-5-5", "claude-sonnet-5-5"
RESULT_TYPES = frozenset({"regex", "file_exists", "llm"})
CONSULT = ("j-arch", "j-cloud", "j-devops", "j-docs", "j-frontend", "j-lang", "j-data")
SKIP = {
    "j-data--effect-orders-schema": {"no-serial"},
    "j-explain--effect-calc-package-explainer": {"self-contained"},
    "j-explain--effect-docs-drift-report": {"self-contained"},
    "architecture-specialist--effect-retry-policy-adr": {"not-retried", "design"},
}
NO_SERIAL_PREFIX = r"^(?:(?!--)[^\n])*?"
EPS = 1e-9

Cases = Dict[str, dict]


def models(asset: str) -> Tuple[str, ...]:
    """Return the models an asset is checked on (writing-plans loads only on Opus)."""
    return (OPUS,) if asset == "writing-plans" else (OPUS, SONNET)


def load(purpose: str, model: str, asset: str, suite: str = "effects") -> Optional[Cases]:
    """Return {case: entry} for one asset batch, reruns applied; None when no complete result exists."""
    root = RESULTS / purpose / "kb" / model / suite
    paths = [root / f"{asset}.json", *sorted(root.glob(f"{asset}@*.json"))]
    found = False
    out: Cases = {}
    for path in paths:
        if not (path.is_file() and path.with_suffix(".key").is_file()):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("partial"):
            continue
        found = True
        for case in doc.get("cases", []):
            out[case["name"]] = case
    return out if found else None


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
        """Fail a case whose kb score drops 1/3 on both models or 2/3 on one."""
        base = {m: load("baseline", m, asset) for m in data}
        drops: Dict[str, Dict[str, float]] = {}
        lines = []
        for model, cases in data.items():
            for name, case in cases.items():
                old = (base.get(model) or {}).get(name)
                if old is None:
                    continue
                kept = self.result_graders(case) & self.result_graders(old) - SKIP.get(name, set())
                if not kept:
                    continue
                b, a = self.score(old, kept), self.score(case, kept)
                if a is None or b is None:
                    continue
                drops.setdefault(name, {})[model] = b - a
                lines.append(f"{name}@{model.split('-')[1]}={b:.2f}->{a:.2f}")
        bad = []
        for name, by in drops.items():
            if any(d >= 2 / 3 - EPS for d in by.values()) or (
                len(by) >= 2 and all(d >= 1 / 3 - EPS for d in by.values())
            ):
                bad.append(name)
        status = "FAIL" if bad else "PASS"
        detail = " ".join(lines) + (f" regressed: {' '.join(bad)}" if bad else "")
        self.emit("no-regression", asset, status, detail)

    @staticmethod
    def result_graders(case: dict) -> set:
        """Return the names of a case's result graders."""
        return {g["name"] for g in case.get("graders", []) if g.get("type") in RESULT_TYPES}

    @staticmethod
    def score(case: dict, names: set) -> Optional[float]:
        """Return the mean over runs of the share of `names` graders that passed."""
        scores = []
        for run in runs(case):
            got = [g for g in run["graders"] if g["name"] in names]
            if got:
                scores.append(sum(bool(g.get("passed")) for g in got) / len(got))
        return sum(scores) / len(scores) if scores else None

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


def own_skill_grader(case: dict, asset: str) -> Optional[str]:
    """Return the case's Skill grader that matches the asset itself, or None."""
    for g in case.get("graders", []):
        cfg = g.get("config") or {}
        if cfg.get("tool") == "Skill" and re.search(
            rf"\b{re.escape(asset)}\"", cfg.get("input_match", "")
        ):
            return g["name"]
    return None


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
                    g = own_skill_grader(case, asset)
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
    args = parser.parse_args()
    if args.cmd == "check":
        return check(args.purpose, args.asset)
    if args.cmd == "figures":
        return figures()
    return spend(args.purpose)


if __name__ == "__main__":
    sys.exit(main())
