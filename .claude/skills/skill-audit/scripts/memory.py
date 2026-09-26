"""Hygiene check for Claude Code auto-memory stores.

Auto-memory is a directory of Markdown files plus a `MEMORY.md` index that the harness loads into
every conversation. Nothing checks it as it grows, so indexes drift from the files they list, links
between memories rot, and an index past the harness's read limit loses its tail silently. This
script checks the mechanical half of that; whether a memory is still true is a judgment it leaves
alone.

Checks, per store:

  FAIL index-missing   the store has memory files but no MEMORY.md
  FAIL index-dangling  an index entry links a file that does not exist inside the store
  FAIL unindexed       a memory file no index entry links (the harness will never surface it)
  FAIL index-too-long  MEMORY.md exceeds 200 lines or 25KB; the harness loads only the first 200
                       lines or 25KB, whichever comes first (code.claude.com/docs/en/memory)
  FAIL frontmatter     a memory file lacks `name`, `description`, or `type`
  FAIL wikilink        a `[[slug]]` names no memory by its `name` or file stem (underscores and
                       hyphens match; `[[slug|alias]]` and `[[slug#heading]]` resolve by slug)
  FAIL unreadable      an entry named `*.md` could not be read as UTF-8 text, or is not a regular
                       file inside the store
  WARN index-line-long an index line exceeds 150 characters

Stores are resolved through symlinks and checked once each, so checkouts that share one store by
symlink report it once. A file that cannot be read is reported and skipped; it never stops the
other stores from being checked. The report names files and checks. A link target is echoed only
when it looks like a slug; anything else is reported by length so memory prose stays out of the
report.

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
from typing import Dict, List, Optional, Sequence, Set

INDEX = "MEMORY.md"
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25 * 1024
MAX_INDEX_LINE_CHARS = 150
REQUIRED_FIELDS = ("name", "description", "type")

INDEX_LINK = re.compile(r"\]\(([^)\s]+\.md)(?:#[^)\s]*)?\)")
WIKILINK = re.compile(r"\[\[([^\[\]\n]+)\]\]")
# `[[...]]` is also TOML array-of-tables and bash test syntax, so code is stripped before matching.
CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
FIELD = re.compile(r"^\s*([A-Za-z_]+):\s*(.*)$")
SLUG_SHAPE = re.compile(r"^[A-Za-z0-9_./-]{1,80}$")


def default_stores() -> List[Path]:
    """Return every auto-memory store under ~/.claude."""
    claude = Path.home() / ".claude"
    stores = sorted((claude / "projects").glob("*/memory"))
    stores += sorted(p for p in (claude / "agent-memory").glob("*") if p.is_dir())
    return [p for p in stores if p.is_dir()]


def frontmatter(text: str) -> Dict[str, str]:
    """Return the frontmatter fields of a memory file, flattening nested keys to their leaf name."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: Dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = FIELD.match(line)
        if match:
            fields.setdefault(match.group(1), match.group(2).strip().strip("\"'"))
    return fields


def slug(value: str) -> str:
    """Normalize a link target so `feedback_a`, `feedback-a`, and `feedback-a|alias` compare equal."""
    value = value.split("|", 1)[0].split("#", 1)[0]
    return value.strip().strip("\"'").lower().replace("_", "-")


def shown(target: str) -> str:
    """Return a link target for the report, or its length when it does not look like a slug."""
    return target if SLUG_SHAPE.match(target) else "<{} chars>".format(len(target))


def finding(level: str, store: str, check: str, file: str, detail: str) -> Dict[str, str]:
    """Build one finding record."""
    return {"level": level, "store": store, "check": check, "file": file, "detail": detail}


def label(path: Path) -> str:
    """Return a store path with the home directory shortened to ~."""
    home = Path.home().resolve()
    resolved = path.resolve()
    if resolved == home or home in resolved.parents:
        return "~/" + resolved.relative_to(home).as_posix()
    return str(path)


def inside(store: Path, path: Path) -> bool:
    """Return whether a path is a regular file whose resolved location sits inside the store."""
    return path.is_file() and store in path.resolve().parents


def read_text(path: Path) -> Optional[str]:
    """Read a UTF-8 text file (BOM tolerated); return None when it cannot be read as such."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None


def check_index(store: Path, name: str, found: List[Dict[str, str]]) -> Optional[Set[str]]:
    """Check MEMORY.md; return the set of resolved paths it links, or None when it has no index."""
    index_path = store / INDEX
    if not index_path.is_file():
        return None
    text = read_text(index_path)
    if text is None:
        found.append(finding("FAIL", name, "unreadable", INDEX, "not a regular UTF-8 text file"))
        return set()
    index_lines = text.splitlines()
    size = len(text.encode("utf-8"))
    if len(index_lines) > MAX_INDEX_LINES or size > MAX_INDEX_BYTES:
        found.append(
            finding(
                "FAIL",
                name,
                "index-too-long",
                INDEX,
                "{} lines, {} bytes; the harness loads only the first {} lines or {} bytes".format(
                    len(index_lines), size, MAX_INDEX_LINES, MAX_INDEX_BYTES
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
                    "line {} is {} chars (max {})".format(number, len(line), MAX_INDEX_LINE_CHARS),
                )
            )
    linked: Set[str] = set()
    targets = {m for line in index_lines for m in INDEX_LINK.findall(line) if "://" not in m}
    for target in sorted(targets):
        if inside(store, store / target):
            linked.add((store / target).resolve().name)
        else:
            found.append(
                finding(
                    "FAIL",
                    name,
                    "index-dangling",
                    shown(target),
                    "linked from MEMORY.md but not a file inside the store",
                )
            )
    return linked


def check_store(store: Path) -> List[Dict[str, str]]:
    """Run every check against one memory store."""
    name = label(store)
    found: List[Dict[str, str]] = []
    entries = sorted(p for p in store.glob("*.md") if p.name != INDEX)
    linked = check_index(store, name, found)

    texts: Dict[str, str] = {}
    for path in entries:
        text = read_text(path) if inside(store, path) else None
        if text is None:
            found.append(
                finding("FAIL", name, "unreadable", path.name, "not a regular UTF-8 text file")
            )
        else:
            texts[path.name] = text

    if linked is None and texts:
        found.append(
            finding(
                "FAIL", name, "index-missing", INDEX, "no MEMORY.md for {} files".format(len(texts))
            )
        )

    names = {slug(frontmatter(text).get("name", "")) for text in texts.values()} - {""}
    names |= {slug(Path(filename).stem) for filename in texts}
    for filename, text in texts.items():
        if linked is not None and filename not in linked:
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
                        "[[{}]] matches no memory name or file".format(shown(target)),
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
    bad = [str(path) for path in requested if not path.is_dir()]
    if bad:
        parser.error("not a directory: " + ", ".join(bad))

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
