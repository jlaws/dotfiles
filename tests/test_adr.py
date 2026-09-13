"""Conformance checks for the on-disk ADR set under docs/adr/.

ADRs are living documents: edited in place when the decision moves, deleted when the decision no
longer exists. Nothing here tolerates a tombstone -- there is no archive directory, no `superseded`
status, and no `supersedes`/`superseded-by` frontmatter, because a dead ADR is removed rather than
kept. See docs/adr/workflow/adrs-are-living-documents.md.
"""

from __future__ import annotations

import datetime as dt
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ADR_ROOT = REPO / "docs" / "adr"
SPEC = REPO / ".claude" / "references" / "architecture" / "architecture-decision-records.md"

# Both match docs/adr/**/*.md and neither is a decision.
SCAFFOLD = {"README.md", "template.md"}

REQUIRED_KEYS = ("status", "topic", "created", "updated", "deciders")

# A replaced decision is edited in place and a dead one is deleted, so neither end of a supersede
# link can exist. A file carrying one is a tombstone that should have been removed.
BANNED_KEYS = ("supersedes", "superseded-by")

LIVE_STATUSES = {"proposed", "accepted"}

# Sections the living-document model retired. Named so a copy-paste from an older ADR fails loudly
# instead of quietly reintroducing an append-only log or a second place to look for alternatives.
RETIRED_SECTIONS = (
    "Amendment Log",
    "Considered Options",
    "Implementation Notes",
    "Related Decisions",
)

HEADING = re.compile(r"^#{1,6}[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Ruled-out ideas are named, never numbered: a table row has no anchor, so "Option 2" stops
# resolving the moment the table is reordered.
NUMBERED_OPTION = re.compile(r"\bOption \d")


def adr_paths() -> list[Path]:
    return sorted(p for p in ADR_ROOT.rglob("*.md") if p.name not in SCAFFOLD)


def frontmatter(text: str) -> dict[str, str]:
    """The YAML block between the leading `---` fences, as raw string values.

    Deliberately not a YAML parser: the frontmatter is five flat scalar keys, and this repo ships
    no runtime pip dependencies, so there is no yaml module to reach for.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def headings(text: str) -> set[str]:
    """Heading names outside fenced code blocks.

    A fenced block in an ADR can hold a shell comment or a markdown example, and neither is a
    section of the record.
    """
    names: set[str] = set()
    fence = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            token = stripped[:3]
            fence = "" if token == fence else (fence or token)
            continue
        if fence:
            continue
        match = HEADING.match(line)
        if match:
            names.add(match.group(1))
    return names


class AdrFrontmatterTests(unittest.TestCase):
    def test_every_adr_declares_the_required_frontmatter(self):
        for path in adr_paths():
            fields = frontmatter(path.read_text())
            rel = path.relative_to(REPO)
            missing = [key for key in REQUIRED_KEYS if key not in fields]
            self.assertFalse(missing, f"{rel}: frontmatter missing {missing}")
            self.assertIn(
                fields["status"],
                LIVE_STATUSES,
                f"{rel}: status {fields['status']!r} not in {sorted(LIVE_STATUSES)}",
            )

    def test_no_adr_carries_a_supersede_link(self):
        for path in adr_paths():
            fields = frontmatter(path.read_text())
            present = [key for key in BANNED_KEYS if key in fields]
            self.assertFalse(
                present,
                f"{path.relative_to(REPO)}: retired frontmatter {present}; a superseded ADR is "
                "deleted, not linked",
            )

    def test_updated_is_never_before_created(self):
        for path in adr_paths():
            fields = frontmatter(path.read_text())
            rel = path.relative_to(REPO)
            for key in ("created", "updated"):
                self.assertRegex(fields[key], ISO_DATE, f"{rel}: {key} is not YYYY-MM-DD")
            self.assertGreaterEqual(
                dt.date.fromisoformat(fields["updated"]),
                dt.date.fromisoformat(fields["created"]),
                f"{rel}: updated precedes created",
            )

    def test_topic_matches_the_containing_directory(self):
        for path in adr_paths():
            fields = frontmatter(path.read_text())
            self.assertEqual(
                fields["topic"],
                path.parent.name,
                f"{path.relative_to(REPO)}: topic field and directory disagree",
            )


class AdrSectionTests(unittest.TestCase):
    def test_no_adr_keeps_a_retired_section(self):
        for path in adr_paths():
            stale = sorted(headings(path.read_text()) & set(RETIRED_SECTIONS))
            self.assertFalse(stale, f"{path.relative_to(REPO)}: retired section(s) {stale}")

    def test_no_adr_cites_an_option_by_number(self):
        for path in adr_paths():
            hits = [
                f"line {n}: {line.strip()}"
                for n, line in enumerate(path.read_text().splitlines(), 1)
                if NUMBERED_OPTION.search(line)
            ]
            self.assertFalse(
                hits,
                f"{path.relative_to(REPO)}: cites an option by number, which `## Ruled Out` rows "
                f"cannot anchor -- name the idea instead: {hits}",
            )


class AdrTemplateTests(unittest.TestCase):
    def test_template_matches_the_standard_template_in_the_spec(self):
        """docs/adr/README.md calls template.md a verbatim copy of the spec's Standard ADR block.

        This is what makes that claim true: edit one and the other has to follow.
        """
        spec = SPEC.read_text()
        self.assertIn(
            "### Standard ADR", spec, f"{SPEC.relative_to(REPO)}: no Standard ADR section"
        )
        after = spec.split("### Standard ADR", 1)[1]
        block = re.search(r"```markdown\n(.*?)```", after, re.DOTALL)
        self.assertIsNotNone(block, "Standard ADR section has no fenced markdown template")
        assert block is not None
        self.assertEqual(
            (ADR_ROOT / "template.md").read_text(),
            block.group(1),
            "docs/adr/template.md has drifted from the Standard ADR template in the spec",
        )


if __name__ == "__main__":
    unittest.main()
