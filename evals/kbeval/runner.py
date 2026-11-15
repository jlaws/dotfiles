"""Budget-safe, resumable batches of `claude plugin eval`.

One batch per asset (`--case '<asset>--*'`), or per case in the workflows suite. A batch is
complete when its JSON exists, is not partial, and its content key matches. Only runs that ended on
an API or rate-limit error rerun, one case at a time; timeouts and turn caps are results. Every
batch is charged to the spend ledger, and none starts whose cap would cross the budget minus a
reserve for in-flight runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

from evals.kbeval import ledger

REPO = Path(__file__).resolve().parents[2]
JUDGE_MODEL = "claude-fable-5-2"
DEFAULT_P95_USD = 5.0
API_ERROR = re.compile(
    r"(?i)\bapi error\b|rate.?limit|overloaded|usage.?limit|\b(?:429|500|502|503|504|529)\b"
)
LOAD_ERROR = re.compile(r"(?i)no eval cases found|invalid case|failed to load|not a trusted plugin")
WRITE_TOOLS = ["Bash", "Write", "Edit"]
P95_LINE = re.compile(r"^Reserve p95: \$(\d+(?:\.\d+)?)", re.MULTILINE)


class PaidCallRefusedError(Exception):
    """Raised when a paid call is attempted without KBEVAL_ALLOW_PAID=1."""


class Cli(Protocol):
    """What the runner needs from the `claude` binary; tests pass a fake."""

    def run(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run `claude <args>`."""
        ...

    def version(self) -> str:
        """Return `claude --version`."""
        ...


class ClaudeCli:
    """The subprocess seam around the `claude` binary."""

    def run(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run `claude <args>`; refuses unless KBEVAL_ALLOW_PAID is exactly "1"."""
        if os.environ.get("KBEVAL_ALLOW_PAID") != "1":
            raise PaidCallRefusedError("paid eval calls need KBEVAL_ALLOW_PAID=1 (set by the make targets)")
        return subprocess.run(
            ["claude", *args], capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False
        )

    def version(self) -> str:
        """Return `claude --version` (free)."""
        result = subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, stdin=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip()


def load_budget(repo: Path = REPO) -> dict:
    """Return the committed budget document."""
    return json.loads((repo / "evals" / "budget.json").read_text(encoding="utf-8"))


def read_p95(repo: Path = REPO) -> float:
    """Return the per-run p95 cost recorded in the README pilot table, or the pre-pilot default."""
    match = P95_LINE.search((repo / "evals" / "README.md").read_text(encoding="utf-8"))
    return float(match.group(1)) if match else DEFAULT_P95_USD


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _runs(doc: dict) -> List[Tuple[str, Optional[str]]]:
    """Return (case name, run error) for every run in both arms."""
    found = []
    for case in doc.get("cases", []):
        for arm in (case.get("arms") or {}).values():
            for run in arm or []:
                found.append((case.get("name", ""), run.get("error")))
    return found


def _api_error_cases(doc: dict) -> List[str]:
    return sorted({name for name, error in _runs(doc) if error and API_ERROR.search(str(error))})


def _content_key(case_dirs: List[Path], parts: List[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode() + b"\0")
    for folder in sorted(case_dirs):
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            digest.update(str(path.relative_to(folder.parent)).encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _complete(json_path: Path, key: str) -> Optional[dict]:
    key_path = json_path.with_suffix(".key")
    if not (json_path.is_file() and key_path.is_file()):
        return None
    if key_path.read_text().strip() != key:
        return None
    doc = json.loads(json_path.read_text())
    return None if doc.get("partial") else doc


def _scan_for_key(paths: List[Path]) -> Optional[Path]:
    for root in paths:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in files:
            try:
                if "sk-ant-" in path.read_text(encoding="utf-8", errors="ignore"):
                    return path
            except OSError:
                continue
    return None


class _Batches:
    """Shared state for one `run` invocation."""

    def __init__(self, suite, staging, model, runs, purpose, jobs, cli, options):
        self.suite = suite
        self.staging = staging
        self.model = model
        self.runs = runs
        self.purpose = purpose
        self.jobs = jobs
        self.cli = cli
        self.plugin: Path = Path(options["plugin"])
        self.state: Path = Path(options.get("state") or ledger.default_path().parent / "results")
        self.ledger_path: Optional[Path] = options.get("ledger_path")
        self.budget: dict = options.get("budget") or load_budget()
        self.p95: float = options.get("p95") if options.get("p95") is not None else read_p95()
        self.cap_limit: Optional[float] = options.get("cap_limit")
        self.invocation_spent = 0.0
        self.version = cli.version()
        self.eval_dir = f"evals-{suite}"

    def cap(self) -> float:
        period = self.budget["period"]
        remaining = float(self.budget["ceilingUsd"]) - ledger.spent(period, self.ledger_path)
        cap = remaining - self.jobs * self.p95
        if self.cap_limit is not None:
            cap = min(cap, self.cap_limit - self.invocation_spent)
        return cap

    def charge(self, label: str, cost: float, charged: str) -> None:
        self.invocation_spent += cost
        ledger.append(
            {
                "period": self.budget["period"],
                "purpose": self.purpose,
                "staging": self.staging,
                "model": self.model,
                "suite": self.suite,
                "batch": label,
                "costUsd": round(cost, 4),
                "charged": charged,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            self.ledger_path,
        )

    def execute(
        self, label: str, case_glob: str, case_dirs: List[Path], json_path: Path, rerun: bool = False
    ) -> Tuple[int, Optional[dict]]:
        """Run one batch; return (status, doc). Status 0 means continue.

        A rerun batch whose stored result still ended on an API error is run again.
        """
        key = _content_key(case_dirs, [self.staging, self.model, JUDGE_MODEL, str(self.runs), self.version])
        doc = _complete(json_path, key)
        if doc is not None and not (rerun and _api_error_cases(doc)):
            return 0, doc
        cap = self.cap()
        if cap <= 0:
            _err(f"budget: batch {label} not started; remaining budget minus reserve is ${cap:.2f}")
            return 2, None
        out_dir = json_path.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "plugin", "eval", str(self.plugin), "--eval-dir", self.eval_dir, "--case", case_glob,
            "--runs", str(self.runs), "--model", self.model, "--judge-model", JUDGE_MODEL,
            "--ablation", "none", "--threshold", "0", "--trust-plugin", "--scaffold", "--no-publish",
            "-j", str(self.jobs), "--json", str(json_path), "--output-dir", str(out_dir),
            "--max-cost-usd", f"{cap:.2f}",
        ]  # fmt: skip
        if self.suite != "triggers":
            args += ["--allow-tools", *WRITE_TOOLS]
        if json_path.exists():
            json_path.unlink()
        result = self.cli.run(args)
        stderr_tail = "\n".join((result.stderr or "").strip().splitlines()[-5:])
        if not json_path.is_file():
            self.charge(label, cap, "cap")
            _err(f"{label}: no JSON result (exit {result.returncode}); charged the ${cap:.2f} cap")
            if stderr_tail:
                _err(stderr_tail)
            return (result.returncode or 1), None
        doc = json.loads(json_path.read_text())
        cost = doc.get("costUsd")
        if _runs(doc) and not cost:
            self.charge(label, cap, "cap")
            _err(f"{label}: result reports runs but cost is {cost!r}; aborting (charged the cap)")
            return 1, None
        self.charge(label, float(cost or 0.0), "actual")
        leaked = _scan_for_key([json_path, out_dir])
        if leaked is not None:
            _err(f"{label}: sk-ant- found in {leaked}; aborting: rotate the key.")
            return 3, None
        if doc.get("partial"):
            reason = doc.get("partialReason")
            _err(f"{label}: partial result ({reason}); stopping")
            return (130 if reason == "interrupted" else 2), None
        if result.returncode == 1 and not _api_error_cases(doc):
            _err(f"{label}: exit 1 with no API-error run; stopping")
            if stderr_tail:
                _err(stderr_tail)
            return 1, None
        json_path.with_suffix(".key").write_text(key + "\n")
        return 0, doc


def _batches(plugin: Path, suite: str, assets: List[str]) -> List[Tuple[str, str, List[Path]]]:
    root = plugin / f"evals-{suite}"
    if not root.is_dir():
        return []
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if suite == "workflows":
        return [(d.name, d.name, [d]) for d in dirs if not assets or d.name in assets]
    found = []
    for asset in assets:
        if "--" in asset:  # an exact case name, batched alone
            found.extend((asset, asset, [d]) for d in dirs if d.name == asset)
            continue
        matching = [d for d in dirs if d.name.startswith(f"{asset}--")]
        if matching:
            found.append((asset, f"{asset}--*", matching))
    return found


def run(
    suite: str,
    staging: str,
    model: str,
    runs: int,
    purpose: str,
    assets: List[str],
    jobs: int = 8,
    cli: Optional[Cli] = None,
    **options,
) -> int:
    """Run one suite for one staging and model, one batch per asset; return an exit status.

    Options: `plugin` (staged plugin dir, required), `state` (results root), `ledger_path`,
    `budget` (dict), `p95` (USD per run), `cap_limit` (USD ceiling for this invocation).
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _err("evals bill an API key; export ANTHROPIC_API_KEY")
        return 1
    batches = _Batches(suite, staging, model, runs, purpose, jobs, cli or ClaudeCli(), options)
    base = batches.state / purpose / staging / model / suite
    for label, case_glob, case_dirs in _batches(batches.plugin, suite, assets):
        status, doc = batches.execute(label, case_glob, case_dirs, base / f"{label}.json")
        if status != 0:
            return status
        by_name: Dict[str, Path] = {d.name: d for d in case_dirs}
        for name in _api_error_cases(doc or {}):
            if name not in by_name:
                continue
            status, _ = batches.execute(
                f"{label}@{name}", name, [by_name[name]], base / f"{label}@{name}.json", rerun=True
            )
            if status != 0:
                return status
    return 0
