"""Adoption report for the Claude knowledge base.

Counts how often each skill, agent, and command is named in Claude Code transcripts, so the orphan
question in SKILL.md is answered with evidence instead of intuition.

This reads the harness's own transcripts, so two properties are structural rather than advisory:

  * It never reads a transcript unless a human asks. `audit.py` never imports it and no build target
    runs it. Its own tests do execute it, but every one of them passes `--transcripts`, so the real
    transcript root is reached only by someone typing the command.
  * It emits counts. Transcript text, session identifiers, and project directory names never reach
    stdout or stderr, because the names of a user's projects are themselves private. An unreadable
    or vanished transcript is counted and skipped, never named in an error.

What it measures is a lower bound. A mention is evidence the asset was in play, not a usage count,
and silence is absence of evidence rather than evidence of absence. The report says so up front.

Usage:
    adoption.py [REPO_ROOT] [--transcripts DIR] [--since DAYS | --all] [--active-days DAYS]
                [--exclude SUBSTRING] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

DEFAULT_TRANSCRIPTS = Path.home() / ".claude" / "projects"
DEFAULT_SINCE_DAYS = 30
DEFAULT_ACTIVE_DAYS = 7

# Transcript directories created by end-to-end test runs. Their traffic is synthetic, so counting it
# would report the test harness's habits as the user's. `--exclude` adds to this list, which is how
# you drop the sessions that edited the knowledge base itself: those name every asset by definition,
# so left in they report editing as use and every asset comes back `active`.
FIXTURE_MARKERS = ("ralph-e2e",)

# Only conversation prose counts, and only from these record types.
#
# Three things that are NOT prose were counted by an earlier version of this script, and each one
# inflated the report on its own:
#
#   * Tool results. The harness stores them as `type: "user"` records carrying `toolUseResult`, so a
#     type filter alone keeps them. Measured 2026-09-12 over 12 transcripts on this machine: 29,701
#     of 82,206 otherwise-countable records were tool results. One `Read` of a SKILL.md credits
#     every asset named inside it.
#   * Tool calls. `tool_use` blocks inside an assistant record carry file paths and arguments.
#   * The record envelope. `cwd` and `gitBranch` sit beside the message, and this repo names branches
#     after the thing being built, so a branch like `feat/subagent-report-contract` credited that
#     asset once per turn for a whole session -- 241 increments, same corpus and date.
#
# So the scan reads `message.content` text blocks and nothing else. The cost is real and is stated in
# LIMITS: an asset invoked only through a tool call, never named in prose, leaves no mention.
COUNTED_TYPES = ("user", "assistant")

# A name counts only as a whole hyphenated token. Without that, every `cmd-j-plan` in a transcript
# would also count as a mention of the `j-plan` command, and every long skill name would inflate the
# shorter one it contains.
#
# The raw line is pre-filtered with this same pattern before anything is parsed: a line that cannot
# hold an asset name is skipped without paying for `json.loads`. A line that survives is parsed and
# then re-matched against its prose alone, which is what makes the envelope and tool-output
# exclusions above real rather than advertised.
#
# That pre-filter is why this is affordable. Measured 2026-09-12: `--all` over 1408 files and
# 1.63 GB takes 1 min 52 s. An earlier draft matched one alternation of every asset name with a
# lookaround on each branch and took just over 5 minutes on a quarter of the data.
TOKEN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")

# `datetime.fromisoformat` on Python 3.9 -- the system interpreter on a fresh Mac -- accepts only 3-
# or 6-digit fractional seconds, and rejects a trailing `Z`. Transcripts use 3 digits today, but a
# harness change to any other width would otherwise zero every windowed count with no diagnostic.
FRACTION = re.compile(r"\.(\d+)")

LIMITS = """What this measures: mentions of an asset's name in conversation prose. That is a
lower bound, and a noisy one. Read a count as evidence the asset was in play, never as a usage count.

- Only prose written by you or the model is counted. Tool calls, tool output, hook output, and the
  catalogue of agents and skills the harness injects on every session are all excluded. So is the
  record envelope, because a branch name matching an asset would otherwise score once per turn.
- The cost of that: an asset invoked only through a tool call, and never named in prose, leaves no
  mention at all. The same goes for a skill whose content was read rather than discussed.
- Transcripts are pruned, so "last seen" is bounded by retention, not by real last use.
- Fixture project directories are skipped, so the counts describe real work only.
- A session that edited the knowledge base names every asset in it. Pass `--exclude` for that
  project, or read a report full of `active` rows as measuring editing rather than use.

`no evidence` means the name was never found. `cold` means it was found, but not recently enough to
clear the active window. Neither means unused."""


def discover_assets(repo: Path) -> "List[Tuple[str, str]]":
    """Return every (kind, name) pair in the Claude tree, sorted."""
    claude = repo / ".claude"
    assets = [("skill", p.parent.name) for p in claude.glob("skills/*/SKILL.md")]
    assets += [("agent", p.stem) for p in claude.glob("agents/*.md")]
    assets += [("command", p.stem) for p in claude.glob("commands/*.md")]
    return sorted(set(assets))


def parse_timestamp(value: object) -> "Optional[datetime]":
    """Parse an ISO-8601 transcript timestamp into an aware UTC datetime, or None."""
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    text = FRACTION.sub(lambda m: "." + m.group(1)[:6].ljust(6, "0"), text, count=1)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def field(record: "Any", *keys: str) -> "Any":
    """Walk a nested key path through a decoded record, returning None if any step is missing."""
    for key in keys:
        if not isinstance(record, dict):
            return None
        record = record.get(key)
    return record


def is_counted(record: "Any") -> bool:
    """Report whether a record is conversation content rather than harness or tool boilerplate."""
    if field(record, "type") not in COUNTED_TYPES:
        return False
    if field(record, "isMeta"):
        return False
    if field(record, "attachment") is not None:
        return False
    # Tool results arrive as ordinary `user` records; this key is what distinguishes them.
    return field(record, "toolUseResult") is None


def prose(record: "Any") -> str:
    """Return only the text a human or the model wrote, dropping tool blocks and the envelope."""
    content = field(record, "message", "content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def project_name(root: Path, path: Path) -> str:
    """Return the project directory a transcript belongs to, whatever depth the file sits at."""
    parts = path.relative_to(root).parts
    return parts[0] if parts else ""


def transcript_files(
    root: Path, cutoff: "Optional[datetime]", exclude: "Sequence[str]"
) -> "Iterator[Tuple[Path, str]]":
    """Yield (transcript, project) pairs worth opening, newest tier included.

    `rglob`, not `glob("*/*.jsonl")`: subagent transcripts live at
    `<project>/<session>/subagents/<file>.jsonl`, and on this machine that tier held 1364 of 1408
    files. A depth-2 glob read 3% of the corpus and still printed "Every transcript".
    """
    for path in sorted(root.rglob("*.jsonl")):
        project = project_name(root, path)
        if any(marker in project for marker in exclude):
            continue
        if cutoff is not None:
            try:
                if path.stat().st_mtime < cutoff.timestamp():
                    continue
            except OSError:
                # Pruned between rglob and stat, or a broken symlink. Never name the path: it is
                # the project directory and session ID the privacy contract forbids emitting.
                continue
        yield path, project


def scan(
    root: Path,
    names: "Sequence[str]",
    cutoff: "Optional[datetime]",
    exclude: "Sequence[str]",
) -> "Tuple[Dict[str, int], Dict[str, datetime], int, int]":
    """Count prose mentions per name, tracking the newest and both file counts."""
    wanted = set(names)
    mentions: "Dict[str, int]" = {name: 0 for name in names}
    last_seen: "Dict[str, datetime]" = {}
    scanned = 0
    unreadable = 0
    for path, _project in transcript_files(root, cutoff, exclude):
        try:
            handle = path.open(encoding="utf-8", errors="replace")
        except OSError:
            # Unreadable, a directory named `*.jsonl`, or pruned mid-scan. Counted, never named.
            unreadable += 1
            continue
        scanned += 1
        with handle:
            for line in handle:
                if "-" not in line:
                    continue
                if not wanted.intersection(TOKEN.findall(line)):
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    # A partially written last line is normal in a live transcript.
                    continue
                if not is_counted(record):
                    continue
                hits = wanted.intersection(TOKEN.findall(prose(record)))
                if not hits:
                    continue
                stamp = parse_timestamp(field(record, "timestamp"))
                if cutoff is not None and (stamp is None or stamp < cutoff):
                    continue
                for name in hits:
                    mentions[name] += 1
                    if stamp is not None:
                        last_seen[name] = max(stamp, last_seen.get(name, stamp))
    return mentions, last_seen, scanned, unreadable


def classify(count: int, seen: "Optional[datetime]", active_days: int) -> str:
    """Label an asset `active`, `cold`, or `no evidence` from its count and recency."""
    if count == 0:
        return "no evidence"
    if seen is None:
        return "cold"
    age = datetime.now(timezone.utc) - seen
    return "active" if age <= timedelta(days=active_days) else "cold"


def build_rows(
    assets: "Sequence[Tuple[str, str]]",
    mentions: "Dict[str, int]",
    last_seen: "Dict[str, datetime]",
    active_days: int,
) -> "List[Dict[str, object]]":
    """Assemble one report row per asset, worst-evidenced last within each kind.

    Counting is keyed by name alone, so if a skill and an agent ever share one, both rows carry the
    combined count. That is honest -- a bare name in prose does not say which was meant -- but it
    means two such rows are one number, not two.
    """
    rows = []
    for kind, name in assets:
        count = mentions.get(name, 0)
        seen = last_seen.get(name)
        rows.append(
            {
                "kind": kind,
                "name": name,
                "mentions": count,
                "last_seen": seen.date().isoformat() if seen else None,
                "status": classify(count, seen, active_days),
            }
        )
    return sorted(rows, key=lambda row: (row["kind"], -int(row["mentions"]), row["name"]))


def cell(value: object) -> str:
    """Escape a value for a markdown table cell, so an odd asset name cannot break the table."""
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render(
    rows: "Sequence[Dict[str, object]]", window: str, scanned: int, unreadable: int, notes: "Sequence[str]"
) -> str:
    """Render the text report: the limits first, then one row per asset."""
    skipped = ", {} unreadable and skipped".format(unreadable) if unreadable else ""
    head = "# KB adoption\n\n{}. Scanned {} transcript file(s){}.\n\n{}\n".format(
        window, scanned, skipped, LIMITS
    )
    if notes:
        head += "\n" + "\n".join("- NOTE: " + note for note in notes) + "\n"
    lines = ["| Asset | Kind | Mentions | Last seen | Status |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                cell(row["name"]),
                cell(row["kind"]),
                row["mentions"],
                cell(row["last_seen"] or "-"),
                cell(row["status"]),
            )
        )
    counts = {
        status: sum(1 for r in rows if r["status"] == status)
        for status in ("active", "cold", "no evidence")
    }
    tail = "\n{} active | {} cold | {} no evidence".format(
        counts["active"], counts["cold"], counts["no evidence"]
    )
    return head + "\n" + "\n".join(lines) + "\n" + tail + "\n"


def emit(payload: "Dict[str, object]", text: str, as_json: bool) -> None:
    """Print either the JSON envelope or the text report, so `--json` holds on every path."""
    print(json.dumps(payload, indent=2) if as_json else text)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser, with the ranges each numeric flag actually accepts."""
    parser = argparse.ArgumentParser(description="Report which KB assets show up in transcripts.")
    parser.add_argument("repo", nargs="?", default=".", help="repository root (default: .)")
    parser.add_argument(
        "--transcripts",
        default=str(DEFAULT_TRANSCRIPTS),
        help="transcript root (default: ~/.claude/projects)",
    )
    parser.add_argument(
        "--since", type=int, default=DEFAULT_SINCE_DAYS, help="window in days, minimum 1 (default: 30)"
    )
    parser.add_argument("--all", action="store_true", help="scan every transcript, ignoring --since")
    parser.add_argument(
        "--active-days",
        type=int,
        default=DEFAULT_ACTIVE_DAYS,
        help="a mention newer than this many days counts as `active`, minimum 0 (default: 7)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="SUBSTRING",
        help="skip project directories whose name contains SUBSTRING (repeatable). Use `=` for a "
        "value starting with a dash: --exclude=-Users-me-Workspace-repo",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    return parser


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    """Parse arguments, scan transcripts, and print the adoption report."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    supplied = list(args.exclude or [])
    if args.since < 1:
        parser.error("--since must be at least 1 day; use --all to ignore the window")
    if args.active_days < 0:
        parser.error("--active-days cannot be negative")
    if any(not marker for marker in supplied):
        parser.error("--exclude needs a non-empty substring; an empty one excludes every project")

    root = Path(args.transcripts).expanduser()
    repo = Path(args.repo).resolve()
    assets = discover_assets(repo)
    names = [name for _, name in assets]
    window = "Every transcript" if args.all else "Window: last {} days".format(args.since)

    if not assets:
        # Symmetry with the transcript-root guard below: a zero report is indistinguishable from a
        # real one, and pointing this at the wrong directory is the likeliest way to get one. The
        # repo path came from the operator, so echoing it discloses nothing they did not type.
        message = "no knowledge-base assets found under {}/.claude".format(repo)
        emit({"error": message, "window": window, "transcripts": 0, "assets": []}, message, args.as_json)
        return 1

    if not root.is_dir():
        # The transcript root is either the harness's own directory or a path the operator passed,
        # so naming it discloses nothing new -- and without it a zero report is undiagnosable.
        message = "no transcripts found at {}".format(root)
        emit({"error": message, "window": window, "transcripts": 0, "assets": []}, message, args.as_json)
        return 0

    cutoff = None if args.all else datetime.now(timezone.utc) - timedelta(days=args.since)
    exclude = list(FIXTURE_MARKERS) + supplied
    mentions, last_seen, scanned, unreadable = scan(root, names, cutoff, exclude)

    # A typo'd --exclude silently excludes nothing, which is the exact failure the flag exists to
    # prevent: the KB-editing sessions stay in and every asset comes back `active`.
    on_disk = [p.name for p in root.iterdir() if p.is_dir()]
    notes = [
        "--exclude {} matched no project directory".format(marker)
        for marker in supplied
        if not any(marker in name for name in on_disk)
    ]

    rows = build_rows(assets, mentions, last_seen, args.active_days)
    payload = {
        "window": window,
        "transcripts": scanned,
        "unreadable": unreadable,
        "notes": notes,
        "assets": rows,
    }
    emit(payload, render(rows, window, scanned, unreadable, notes), args.as_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
