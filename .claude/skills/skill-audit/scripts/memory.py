"""Hygiene check for Claude Code auto-memory stores.

Auto-memory is a directory of Markdown files plus a `MEMORY.md` index that the harness loads into
every conversation. Nothing checks it as it grows, so indexes drift from the files they list, links
between memories rot, and an index past the harness's line cap loses its tail silently. This script
checks the mechanical half of that; whether a memory is still true is a judgment it leaves alone.

Checks, per store:

  FAIL index-missing   the store has memory files but no MEMORY.md
  FAIL index-dangling  an index entry links a file that does not exist
  FAIL unindexed       a memory file no index entry links (the harness will never surface it)
  FAIL index-too-long  MEMORY.md exceeds 200 lines; the harness truncates everything after that
  FAIL frontmatter     a memory file lacks `name`, `description`, or `type`
  FAIL wikilink        a `[[slug]]` names no memory by its `name` or file stem (underscores and
                       hyphens match)
  WARN index-line-long an index line exceeds 150 characters

Stores are resolved through symlinks and checked once each, so checkouts that share one store by
symlink report it once. The report names files and checks, never memory contents.

Usage:
    memory.py [--memory-dir DIR ...] [--json]

With no `--memory-dir`, every `~/.claude/projects/*/memory` and `~/.claude/agent-memory/*` store is
checked. Exit 0 when nothing FAILs, 1 when something does, 2 on a bad argument.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

INDEX = "MEMORY.md"
MAX_INDEX_LINES = 200
MAX_INDEX_LINE_CHARS = 150
REQUIRED_FIELDS = ("name", "description", "type")

INDEX_LINK = re.compile(r"\]\(([^)\s]+\.md)\)")
WIKILINK = re.compile(r"\[\[([^\]\n]+)\]\]")
# `[[...]]` is also TOML array-of-tables and bash test syntax, so code is stripped before matching.
CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
FIELD = re.compile(r"^\s*([A-Za-z_]+):\s*(.*)$")


def default_stores() -> List[Path]:
    """Return every auto-memory store under ~/.claude."""
    claude = Path.home() / ".claude"
    stores = sorted((claude / "projects").glob("*/memory"))
    stores += sorted(p for p in (claude / "agent-memory").glob("*") if p.is_dir())
    return [p for p in stores if p.is_dir()]


def frontmatter(text: str) -> Dict[str, str]:
    """Return the frontmatter fields of a memory file, flattening one level of nesting."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: Dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = FIELD.match(line)
        if match:
            fields.setdefault(match.group(1), match.group(2).strip())
    return fields


def slug(value: str) -> str:
    """Normalize a memory name so `feedback_a` and `feedback-a` compare equal."""
    return value.strip().strip("\"'").lower().replace("_", "-")


def finding(level: str, store: str, check: str, file: str, detail: str) -> Dict[str, str]:
    """Build one finding record."""
    return {"level": level, "store": store, "check": check, "file": file, "detail": detail}


def label(path: Path) -> str:
    """Return a store path with the home directory shortened to ~."""
    home = str(Path.home())
    text = str(path)
    return "~" + text[len(home) :] if text.startswith(home) else text


def check_store(store: Path) -> List[Dict[str, str]]:
    """Run every check against one memory store."""
    name = label(store)
    files = sorted(p for p in store.glob("*.md") if p.name != INDEX)
    found: List[Dict[str, str]] = []
    index_path = store / INDEX

    if not index_path.is_file():
        if files:
            found.append(
                finding(
                    "FAIL",
                    name,
                    "index-missing",
                    INDEX,
                    "no MEMORY.md for {} files".format(len(files)),
                )
            )
        linked = set()
    else:
        index_lines = index_path.read_text(encoding="utf-8").splitlines()
        if len(index_lines) > MAX_INDEX_LINES:
            found.append(
                finding(
                    "FAIL",
                    name,
                    "index-too-long",
                    INDEX,
                    "{} lines; the harness drops everything after line {}".format(
                        len(index_lines), MAX_INDEX_LINES
                    ),
                )
            )
        for number, line in enumerate(index_lines, 1):
            if len(line) > MAX_INDEX_LINE_CHARS:
                found.append(
                    finding(
                        "WARN",
                        name,
                        "index-line-long",
                        INDEX,
                        "line {} is {} chars (max {})".format(
                            number, len(line), MAX_INDEX_LINE_CHARS
                        ),
                    )
                )
        linked = {m for line in index_lines for m in INDEX_LINK.findall(line)}
        for target in sorted(linked):
            if not (store / target).is_file():
                found.append(
                    finding(
                        "FAIL", name, "index-dangling", target, "linked from MEMORY.md but missing"
                    )
                )

    texts = {p.name: p.read_text(encoding="utf-8") for p in files}
    names = {slug(frontmatter(text).get("name", "")) for text in texts.values()} - {""}
    names |= {slug(Path(filename).stem) for filename in texts}
    for filename, text in texts.items():
        if index_path.is_file() and filename not in linked:
            found.append(
                finding("FAIL", name, "unindexed", filename, "no MEMORY.md entry links it")
            )
        fields = frontmatter(text)
        missing = [field for field in REQUIRED_FIELDS if not fields.get(field)]
        if missing:
            found.append(
                finding("FAIL", name, "frontmatter", filename, "missing " + ", ".join(missing))
            )
        for target in WIKILINK.findall(CODE.sub("", text)):
            if slug(target) not in names:
                found.append(
                    finding(
                        "FAIL",
                        name,
                        "wikilink",
                        filename,
                        "[[{}]] matches no memory name or file".format(target),
                    )
                )
    return found


def render(findings: List[Dict[str, str]], stores: int) -> str:
    """Return the human-readable report."""
    lines = [
        "{level} {store} {check} {file}: {detail}".format(**f)
        for f in sorted(findings, key=lambda f: (f["level"], f["store"], f["check"], f["file"]))
    ]
    fails = sum(1 for f in findings if f["level"] == "FAIL")
    warns = len(findings) - fails
    lines.append("{} store(s) checked: {} FAIL, {} WARN".format(stores, fails, warns))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Check Claude Code auto-memory stores for index and link rot."
    )
    parser.add_argument(
        "--memory-dir",
        action="append",
        default=None,
        metavar="DIR",
        help="a memory store to check (repeatable); default: every store under ~/.claude",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, check each store once, and print the report."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    requested = (
        [Path(d).expanduser() for d in args.memory_dir] if args.memory_dir else default_stores()
    )
    for path in requested:
        if not path.is_dir():
            parser.error("{} is not a directory".format(path))

    stores: List[Path] = []
    for path in requested:
        resolved = path.resolve()
        if resolved not in stores:
            stores.append(resolved)

    findings = [f for store in stores for f in check_store(store)]
    if args.as_json:
        print(json.dumps({"stores": [label(s) for s in stores], "findings": findings}, indent=2))
    else:
        print(render(findings, len(stores)))
    return 1 if any(f["level"] == "FAIL" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
