"""Adoption report for the Claude knowledge base.

Counts how often each skill, agent, and command is named in Claude Code transcripts, so the orphan
question in SKILL.md is answered with evidence instead of intuition.

This reads the harness's own transcripts, so two properties are structural rather than advisory:

  * It is never automatic. No Makefile target calls it, `audit.py` never imports it, and its only
    caller is the `skill-audit` skill, reached by typing the command.
  * It emits counts. Transcript text, session identifiers, and project directory names never reach
    stdout, because the names of a user's projects are themselves private.

What it measures is a lower bound. A mention is evidence the asset was in play, not a usage count,
and silence is absence of evidence rather than evidence of absence. The report says so up front.

Usage:
    python3 .claude/skills/skill-audit/scripts/adoption.py [REPO_ROOT] [--since DAYS] [--all]
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

# Only conversation turns count. The harness injects its own catalogue of every agent and skill into
# attachment records and into the meta records that carry a command body, so a scan that counts every
# record reports the catalogue rather than the session. Measured over 40 transcripts outside this
# repo: counting every record named 79 of 79 assets and left nothing silent, while counting only
# conversation turns named 49 and left 30 with no evidence.
COUNTED_TYPES = ("user", "assistant")

# A name counts only as a whole hyphenated token. Without that, every `cmd-j-plan` in a transcript
# would also count as a mention of the `j-plan` command, and every long skill name would inflate the
# shorter one it contains.
#
# Extracting tokens and intersecting beats one alternation of all ~79 names with lookarounds on each
# branch: over the real transcript directory that alternation took just over 5 minutes, which is too
# slow for something a human runs and waits on. Every asset name is hyphenated, so a line with no
# hyphen cannot hold one and is rejected before the regex runs at all.
TOKEN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")

LIMITS = """What this measures: mentions of an asset's name in transcripts. That is a lower bound, and
a noisy one. Read a count as evidence the asset was in play, never as a usage count.

- Only conversation turns are counted. Hook output, and the catalogue of agents and skills the
  harness injects on every session, are excluded rather than counted; left in, they make every
  asset look active no matter what the session did.
- A skill whose content was read rather than invoked can leave no mention at all.
- Transcripts are pruned, so "last seen" is bounded by retention, not by real last use.
- Fixture project directories are skipped, so the counts describe real work only.
- A session that edited the knowledge base names every asset in it. Pass `--exclude` for that
  project, or read a report full of `active` rows as measuring editing rather than use.

`cold` and `no evidence` both mean no evidence was found. Neither means unused."""


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
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def field(record: "Any", *keys: str) -> "Any":
    """Walk a nested key path through a decoded record, returning None if any step is missing."""
    for key in keys:
        if not isinstance(record, dict):
            return None
        record = dict(record).get(key)
    return record


def is_counted(record: "Any") -> bool:
    """Report whether a record is conversation content rather than harness boilerplate."""
    if field(record, "type") not in COUNTED_TYPES:
        return False
    if field(record, "isMeta"):
        return False
    return field(record, "attachment") is None


def transcript_files(
    root: Path, cutoff: "Optional[datetime]", exclude: "Sequence[str]"
) -> "Iterator[Path]":
    """Yield transcripts worth opening, skipping excluded projects and files older than the window."""
    for path in sorted(root.glob("*/*.jsonl")):
        if any(marker in path.parent.name for marker in exclude):
            continue
        if cutoff is not None and path.stat().st_mtime < cutoff.timestamp():
            continue
        yield path


def scan(
    root: Path,
    names: "Sequence[str]",
    cutoff: "Optional[datetime]",
    exclude: "Sequence[str]",
) -> "Tuple[Dict[str, int], Dict[str, datetime], int]":
    """Count whole-token mentions per name and track the newest one, returning the file count too."""
    wanted = set(names)
    mentions: "Dict[str, int]" = {name: 0 for name in names}
    last_seen: "Dict[str, datetime]" = {}
    scanned = 0
    for path in transcript_files(root, cutoff, exclude):
        scanned += 1
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "-" not in line:
                    continue
                hits = wanted.intersection(TOKEN.findall(line))
                if not hits:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    # A partially written last line is normal in a live transcript.
                    record = None
                if not is_counted(record):
                    continue
                stamp = parse_timestamp(field(record, "timestamp"))
                if cutoff is not None and (stamp is None or stamp < cutoff):
                    continue
                for name in hits:
                    mentions[name] += 1
                    if stamp and stamp > last_seen.get(name, stamp - timedelta(seconds=1)):
                        last_seen[name] = stamp
    return mentions, last_seen, scanned


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
    """Assemble one report row per asset, worst-evidenced last within each kind."""
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


def render(rows: "Sequence[Dict[str, object]]", window: str, scanned: int) -> str:
    """Render the text report: the limits first, then one row per asset."""
    head = "# KB adoption\n\n{}. Scanned {} transcript file(s).\n\n{}\n".format(window, scanned, LIMITS)
    lines = ["| Asset | Kind | Mentions | Last seen | Status |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                row["name"], row["kind"], row["mentions"], row["last_seen"] or "-", row["status"]
            )
        )
    counts = {status: sum(1 for r in rows if r["status"] == status) for status in ("active", "cold", "no evidence")}
    tail = "\n{} active | {} cold | {} no evidence".format(
        counts["active"], counts["cold"], counts["no evidence"]
    )
    return head + "\n" + "\n".join(lines) + "\n" + tail + "\n"


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    """Parse arguments, scan transcripts, and print the adoption report."""
    parser = argparse.ArgumentParser(description="Report which KB assets show up in transcripts.")
    parser.add_argument("repo", nargs="?", default=".", help="repository root")
    parser.add_argument("--transcripts", default=str(DEFAULT_TRANSCRIPTS), help="transcript root")
    parser.add_argument("--since", type=int, default=DEFAULT_SINCE_DAYS, help="window in days")
    parser.add_argument("--all", action="store_true", help="scan every transcript, ignoring --since")
    parser.add_argument(
        "--active-days", type=int, default=DEFAULT_ACTIVE_DAYS, help="recency cutoff for `active`"
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SUBSTRING",
        help="skip project directories whose name contains SUBSTRING (repeatable)",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(args.transcripts).expanduser()
    assets = discover_assets(Path(args.repo).resolve())
    names = [name for _, name in assets]

    if not root.is_dir():
        # The transcript root is the harness's own directory, not a user project, so naming it is
        # the one path this script may print -- without it a zero report is undiagnosable.
        print("no transcripts found at {}".format(root))
        return 0

    cutoff = None if args.all else datetime.now(timezone.utc) - timedelta(days=args.since)
    window = "Every transcript" if args.all else "Window: last {} days".format(args.since)
    exclude = list(FIXTURE_MARKERS) + list(args.exclude)
    mentions, last_seen, scanned = scan(root, names, cutoff, exclude)
    rows = build_rows(assets, mentions, last_seen, args.active_days)

    if args.as_json:
        print(json.dumps({"window": window, "transcripts": scanned, "assets": rows}, indent=2))
    else:
        print(render(rows, window, scanned))
    return 0


if __name__ == "__main__":
    sys.exit(main())
