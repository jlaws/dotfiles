"""Budget-safe, resumable batches of `claude plugin eval`.

One batch per asset (`--case '<asset>--*'`), or per case in the workflows suite. A batch is
complete when its JSON exists, is not partial, and its content key matches; the key covers the
staged case dirs and the staged KB (skills, commands, agents, references, fixtures). Only runs that
ended on an API or rate-limit error rerun, one case at a time; timeouts and turn caps are results.

Spend: each batch's cap is the smallest of the remaining budget minus a reserve for in-flight runs,
a per-batch allowance (cases x runs x a per-run allowance), and any invocation cap. The cap is
reserved in the ledger before the paid call and settled to the actual cost after it, so a crash or
interrupt leaves the cap charged and concurrent runners cannot spend the same headroom.

One CLI per invocation: `ClaudeCli` resolves `claude` to its real path once and runs that binary
throughout, and each batch checks the version before its paid call. A change stops the run with
exit 4; a result whose own `claudeVersion` differs from the step's is settled, then moved to
`quarantine/` beside the results with a README.txt note. An interrupt, and any other path that
keeps the cap charged, prints the `settle` command that closes its reservation once the real cost is
known; nothing releases it automatically.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

from evals.kbeval import ledger
from evals.kbeval.results import SIDECAR

REPO = Path(__file__).resolve().parents[2]
JUDGE_MODEL = "claude-fable-5-2"
DEFAULT_P95_USD = 5.0
# A run is allowed this many p95s, with a floor, before the CLI's own ceiling stops the batch.
RUN_ALLOWANCE_P95S = 10
RUN_ALLOWANCE_FLOOR_USD = 5.0
MIN_CAP_USD = 0.01
API_ERROR = re.compile(
    r"(?i)\bapi error\b|rate.?limit|overloaded|usage.?limit|\bstatus(?: code)?:? ?(?:429|5\d\d)\b"
    r"|\b(?:429|529)\b|econnreset|etimedout|connection (?:reset|refused)"
)
TURN_CAP = re.compile(r"Reached maximum number of turns")
TIMEOUT = re.compile(r"timed out after")
LOAD_ERROR = re.compile(r"(?i)no eval cases found|invalid case|failed to load|not a trusted plugin")
WRITE_TOOLS = ["Bash", "Write", "Edit"]
VERSION_STOP = 4


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
    """The subprocess seam around the `claude` binary, pinned to one real path per instance."""

    def __init__(self, binary: Optional[str] = None):
        """Pin `binary`, or resolve `claude` from PATH on first use."""
        self._binary = binary

    @property
    def binary(self) -> str:
        """Return the real path of `claude`, resolved on first use and then kept."""
        if self._binary is None:
            found = shutil.which("claude")
            if found is None:
                raise FileNotFoundError("claude not found on PATH")
            self._binary = os.path.realpath(found)
        return self._binary

    def run(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run `claude <args>`; refuses unless KBEVAL_ALLOW_PAID is exactly "1"."""
        if os.environ.get("KBEVAL_ALLOW_PAID") != "1":
            raise PaidCallRefusedError(
                "paid eval calls need KBEVAL_ALLOW_PAID=1 (set by the make targets)"
            )
        return subprocess.run(
            [self.binary, *args],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=False,
        )

    def version(self) -> str:
        """Return `claude --version` (free)."""
        result = subprocess.run(
            [self.binary, "--version"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip()


def cli_version(raw: str) -> str:
    """Return the leading dotted number of `claude --version` output, else the stripped text."""
    match = re.match(r"\s*(\d+(?:\.\d+)+)", raw)
    return match.group(1) if match else raw.strip()


def load_budget(repo: Optional[Path] = None) -> dict:
    """Return the committed budget document."""
    return json.loads(((repo or REPO) / "evals" / "budget.json").read_text(encoding="utf-8"))


def judge_family(model: str) -> str:
    """Return the family of a Claude model id (`claude-<family>-...`), else the id itself."""
    match = re.match(r"^claude-([a-z]+)-", model)
    return match.group(1) if match else model


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _say(message: str) -> None:
    print(message, flush=True)


def _runs(doc: dict) -> List[Tuple[str, Optional[str]]]:
    """Return (case name, run error) for every run in both arms."""
    found = []
    for case in doc.get("cases", []):
        for arm in (case.get("arms") or {}).values():
            for run in arm or []:
                found.append((case.get("name", ""), run.get("error")))
    return found


def _trace_paths(doc: dict) -> List[str]:
    """Return every run's kept trace path (present only under --keep-temp)."""
    return [
        run["tracePath"]
        for case in doc.get("cases", [])
        for arm in (case.get("arms") or {}).values()
        for run in arm or []
        if run.get("tracePath")
    ]


def _api_error_cases(doc: dict) -> List[str]:
    return sorted({name for name, error in _runs(doc) if error and API_ERROR.search(str(error))})


def _tree_digest(root: Path, skip_prefix: str = "") -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if skip_prefix and rel.parts[0].startswith(skip_prefix):
            continue
        digest.update(str(rel).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _content_key(case_dirs: List[Path], parts: List[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode() + b"\0")
    for folder in sorted(case_dirs):
        digest.update(folder.name.encode() + b"\0" + _tree_digest(folder).encode() + b"\0")
    return digest.hexdigest()


def _read_doc(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _complete(json_path: Path, key: str) -> Optional[dict]:
    key_path = json_path.with_suffix(".key")
    if not (json_path.is_file() and key_path.is_file()):
        return None
    if key_path.read_text(encoding="utf-8").strip() != key:
        return None
    doc = _read_doc(json_path)
    return None if doc is None or doc.get("partial") else doc


def _scan_for_key(paths: List[Path]) -> Optional[Path]:
    for root in paths:
        if not root.exists():
            continue
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in files:
            try:
                if "sk-ant-" in path.read_text(encoding="utf-8", errors="ignore"):
                    return path
            except OSError:
                continue
    return None


def _write_sidecar(json_path: Path, kb_digest: str) -> None:
    json_path.with_suffix(SIDECAR).write_text(
        json.dumps({"kbDigest": kb_digest}) + "\n", encoding="utf-8"
    )


def _discard(json_path: Path) -> None:
    for path in (json_path, json_path.with_suffix(".key"), json_path.with_suffix(SIDECAR)):
        if path.exists():
            path.unlink()


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
        p95 = options.get("p95")
        self.p95: float = float(
            p95 if p95 is not None else self.budget.get("p95RunUsd", DEFAULT_P95_USD)
        )
        self.cap_limit: Optional[float] = options.get("cap_limit")
        self.keep_temp: bool = bool(options.get("keep_temp"))
        self.invocation_spent = 0.0
        self.version = cli.version()
        self.version_number = cli_version(self.version)
        self.eval_dir = f"evals-{suite}"
        # Everything staged outside the eval dirs: skills, commands, agents, references, fixtures,
        # CLAUDE.md, and harness.json.
        self.kb_digest = _tree_digest(self.plugin, skip_prefix="evals-")

    @property
    def reserve_usd(self) -> float:
        return self.jobs * self.p95

    def cap(self, cases: int) -> Tuple[float, str]:
        """Return the batch cap and which limit bound it."""
        ceiling = float(self.budget["ceilingUsd"])
        spent = ledger.spent(self.budget["period"], self.ledger_path)
        allowance = cases * self.runs * max(RUN_ALLOWANCE_P95S * self.p95, RUN_ALLOWANCE_FLOOR_USD)
        limits = [
            (ceiling - spent - self.reserve_usd,
             f"budget (ceiling ${ceiling:.2f}, spent ${spent:.2f}, reserve ${self.reserve_usd:.2f})"),
            (allowance, f"batch allowance ({cases} cases x {self.runs} runs)"),
        ]  # fmt: skip
        if self.cap_limit is not None:
            limits.append(
                (self.cap_limit - self.invocation_spent, f"invocation cap ${self.cap_limit:.2f}")
            )
        return min(limits, key=lambda item: item[0])

    def _entry(self, label: str, cost: float, charged: str, **extra) -> dict:
        return {
            "period": self.budget["period"],
            "purpose": self.purpose,
            "staging": self.staging,
            "model": self.model,
            "suite": self.suite,
            "batch": label,
            "costUsd": round(cost, 4),
            "charged": charged,
            "claudeVersion": self.version,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **extra,
        }

    def settle(self, label: str, cap: float, actual: float, reserved: dict, **extra) -> None:
        """Replace a reserved cap with the actual cost (a negative settle refunds the difference).

        Skipped, with a note, when a row already settles this reservation.
        """
        entry = self._entry(
            label, actual - cap, "settle", actualUsd=round(actual, 4),
            settles=reserved.get("id") or reserved.get("ts"), **extra,
        )  # fmt: skip
        if ledger.settle_id(entry, self.ledger_path):
            self.invocation_spent += actual
        else:
            held = entry["settles"]
            by_hand = next(
                (e for e in ledger.entries(self.ledger_path) if e.get("settles") == held), {}
            )
            self.invocation_spent += float(by_hand.get("actualUsd", cap))
            _err(f"{label}: reservation already settled by hand; that row stands")

    def settle_command(self, label: str, reserved: dict) -> str:
        """Return the command that settles exactly this reservation by hand."""
        values = (self.purpose, self.staging, self.model, self.suite, label)
        purpose, staging, model, suite, batch = (shlex.quote(v) for v in values)
        reservation = shlex.quote(str(reserved.get("id") or reserved.get("ts")))
        return (
            f"python3 -m evals.kbeval settle --purpose {purpose} --staging {staging} "
            f"--model {model} --suite {suite} --batch {batch} --id {reservation} --actual-usd N"
        )

    def charged_hint(self, label: str, cap: float, reserved: dict) -> str:
        """Return the line that tells the owner how to close a reservation left at its cap."""
        return (
            f"{label}: the ${cap:.2f} reservation stays charged. Once the real cost is known, "
            f"settle it (replace N with the billed USD):\n  {self.settle_command(label, reserved)}"
        )

    def quarantine(
        self, label: str, json_path: Path, out_dir: Path, found: str, actual: float
    ) -> Path:
        """Move a result written by another CLI version aside with a README.txt note."""
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rel = json_path.parent.relative_to(self.state)
        base = self.state.parent / "quarantine"
        found = re.sub(r"[^0-9A-Za-z.]", "_", found)
        folder = base / f"{stamp}-cli-{found}"
        number = 2
        while (folder / rel / json_path.name).exists() or (folder / rel / out_dir.name).exists():
            folder = base / f"{stamp}-cli-{found}-{number}"
            number += 1
        target = folder / rel
        target.mkdir(parents=True, exist_ok=True)
        moved = []
        if json_path.exists():
            shutil.move(str(json_path), str(target / json_path.name))
            moved.append(json_path.name)
        if out_dir.exists():
            shutil.move(str(out_dir), str(target / out_dir.name))
            moved.append(f"{out_dir.name}/")
        binary = getattr(self.cli, "binary", "claude")
        with open(folder / "README.txt", "a", encoding="utf-8") as note:
            note.write(
                f"batch: {self.purpose}/{self.staging}/{self.model}/{self.suite}/{label}\n"
                f"step claude version: {self.version_number}\n"
                f"result claude version: {found}\n"
                f"binary: {binary}\n"
                f"moved files: {', '.join(moved)}\n"
                f"settled: ${actual:.2f}\n\n"
            )
        return target

    def execute(
        self,
        label: str,
        case_glob: str,
        case_dirs: List[Path],
        json_path: Path,
        rerun: bool = False,
    ) -> Tuple[int, Optional[dict]]:
        """Run one batch; return (status, doc). Status 0 means continue.

        A rerun batch whose stored result still ended on an API error is run again. A main batch
        that runs again discards its old per-case rerun results.
        """
        parts = [
            self.staging,
            self.model,
            JUDGE_MODEL,
            str(self.runs),
            self.version,
            self.kb_digest,
        ]
        key = _content_key(case_dirs, parts)
        doc = _complete(json_path, key)
        if doc is not None and not (rerun and _api_error_cases(doc)):
            _say(f"{label}: complete, skipped")
            _write_sidecar(json_path, self.kb_digest)
            return 0, doc
        now = cli_version(self.cli.version())
        if now != self.version_number:
            _err(
                f"{label}: claude version changed from {self.version_number} to {now}; "
                "stopping before the paid call"
            )
            return VERSION_STOP, None
        cap, bound = self.cap(len(case_dirs))
        if cap < MIN_CAP_USD:
            _err(f"budget: batch {label} not started; cap ${cap:.2f} is bound by {bound}")
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
        if self.keep_temp:
            args.append("--keep-temp")
        if self.suite != "triggers":
            args += ["--allow-tools", *WRITE_TOOLS]
        _discard(json_path)
        if not rerun:
            for stale in json_path.parent.glob(f"{label}@*.json"):
                _discard(stale)
        reserved = ledger.reserve(
            self._entry(label, cap, "reserved", capUsd=round(cap, 4)),
            float(self.budget["ceilingUsd"]), self.reserve_usd, self.ledger_path,
        )  # fmt: skip
        if not reserved:
            _err(f"budget: batch {label} not started; another runner took the headroom")
            return 2, None
        _say(f"{label}: started ({len(case_dirs)} cases, cap ${cap:.2f}, bound by {bound})")
        try:
            result = self.cli.run(args)
        except KeyboardInterrupt:
            _err(f"{label}: interrupted.\n{self.charged_hint(label, cap, reserved)}")
            raise
        log = out_dir / "cli.log"
        log.write_text(
            f"exit {result.returncode}\n--- stdout\n{result.stdout or ''}\n--- stderr\n{result.stderr or ''}\n",
            encoding="utf-8",
        )
        doc = _read_doc(json_path) if json_path.is_file() else None
        traces = [Path(p) for p in _trace_paths(doc or {}) if Path(p).is_file()]
        leaked = _scan_for_key([json_path, out_dir, *traces])
        if leaked is not None:
            if json_path.exists():
                json_path.rename(json_path.with_suffix(".quarantined"))
            _err(
                f"{label}: sk-ant- found in {leaked}; result quarantined; aborting: rotate the key.\n"
                f"{self.charged_hint(label, cap, reserved)}"
            )
            return 3, None
        stderr_tail = "\n".join((result.stderr or "").strip().splitlines()[-5:])
        if doc is None:
            if LOAD_ERROR.search(result.stderr or ""):
                self.settle(
                    label, cap, 0.0, reserved, exitCode=result.returncode, reason="load error"
                )
                _err(
                    f"{label}: the eval did not load (exit {result.returncode}); charged $0. Log: {log}"
                )
                _err(stderr_tail)
                return 1, None
            _err(
                f"{label}: no readable JSON result (exit {result.returncode}); "
                f"Log: {log}\n{self.charged_hint(label, cap, reserved)}"
            )
            if stderr_tail:
                _err(stderr_tail)
            return (result.returncode or 1), None
        raw = doc.get("costUsd")
        cost = (
            float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else math.nan
        )
        if not math.isfinite(cost) or cost < 0 or (_runs(doc) and not cost):
            _err(
                f"{label}: result reports cost {raw!r}; aborting\n"
                f"{self.charged_hint(label, cap, reserved)}"
            )
            return 1, None
        reason = doc.get("partialReason") if doc.get("partial") else None
        self.settle(
            label, cap, cost, reserved, exitCode=result.returncode, runCount=len(_runs(doc)),
            partial=bool(doc.get("partial")), reason=reason, overCap=cost > cap,
        )  # fmt: skip
        spent = ledger.spent(self.budget["period"], self.ledger_path)
        _say(f"{label}: done, ${cost:.2f}; period spent ${spent:.2f}")
        found = doc.get("claudeVersion")
        if not found:
            _err(f"{label}: result carries no claudeVersion; kept, but its version is unchecked")
        elif cli_version(str(found)) != self.version_number:
            target = self.quarantine(label, json_path, out_dir, cli_version(str(found)), cost)
            _err(
                f"{label}: result written by claude {found}, not {self.version_number}; "
                f"settled ${cost:.2f} and moved to {target}"
            )
            return VERSION_STOP, None
        if doc.get("partial"):
            _err(f"{label}: partial result ({reason}); stopping. Log: {log}")
            return (130 if reason == "interrupted" else 2), None
        api_errors = _api_error_cases(doc)
        if result.returncode not in (0, 1) or (result.returncode == 1 and not api_errors):
            _err(
                f"{label}: exit {result.returncode} without an API-error run; stopping. Log: {log}"
            )
            if stderr_tail:
                _err(stderr_tail)
            return result.returncode, None
        if api_errors:
            _err(f"{label}: API errors in {', '.join(api_errors)}; rerunning those cases")
        json_path.with_suffix(".key").write_text(key + "\n", encoding="utf-8")
        _write_sidecar(json_path, self.kb_digest)
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
    `budget` (dict), `p95` (USD per run; default `budget["p95RunUsd"]`), `cap_limit` (USD ceiling
    for this invocation), `keep_temp` (keep each run's temp dir so its trace can be read; smoke
    sets it).
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _err("evals bill an API key; export ANTHROPIC_API_KEY")
        return 1
    if not re.match(r"^claude-[a-z]+-", model):
        _err(
            f"model {model} is not a full model id (claude-<family>-...); the judge-family guard cannot read it"
        )
        return 1
    if judge_family(model) == judge_family(JUDGE_MODEL):
        _err(f"model {model} is in the judge's family ({JUDGE_MODEL}); a run cannot grade itself")
        return 1
    if jobs < 1 or runs < 1:
        _err(f"jobs and runs must be at least 1 (got jobs={jobs}, runs={runs})")
        return 1
    cli = cli or ClaudeCli()
    batches = _Batches(suite, staging, model, runs, purpose, jobs, cli, options)
    _say(f"claude: {getattr(cli, 'binary', 'claude')} ({batches.version})")
    base = batches.state / purpose / staging / model / suite
    for label, case_glob, case_dirs in _batches(batches.plugin, suite, assets):
        status, doc = batches.execute(label, case_glob, case_dirs, base / f"{label}.json")
        if status != 0:
            return status
        by_name: Dict[str, Path] = {d.name: d for d in case_dirs}
        for name in _api_error_cases(doc or {}):
            if name not in by_name:
                continue
            status, rerun_doc = batches.execute(
                f"{label}@{name}", name, [by_name[name]], base / f"{label}@{name}.json", rerun=True
            )
            if status != 0:
                return status
            if rerun_doc is not None and _api_error_cases(rerun_doc):
                _err(f"{label}@{name}: still an API error; rerun the same command later")
    return 0
