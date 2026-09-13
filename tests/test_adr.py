"""Conformance checks for the on-disk ADR set under docs/adr/.

ADRs are living documents: edited in place when the decision moves, deleted when the decision no
longer exists. Nothing here tolerates a tombstone -- there is no archive directory, no `superseded`
status, and no `supersedes`/`superseded-by` frontmatter, because a dead ADR is removed rather than
kept. See docs/adr/workflow/adrs-are-living-documents.md.

Deleting a record is legal under that model, so every check that walks the ADR set asserts the walk
found something first. Without that, emptying `docs/adr/` turns this file green instead of red.
"""

from __future__ import annotations

import datetime as dt
import re
import unittest
from pathlib import Path

from tests.markdown import has_unterminated_fence, heading_texts, lines_outside_fences

REPO = Path(__file__).resolve().parents[1]
ADR_ROOT = REPO / "docs" / "adr"
README = ADR_ROOT / "README.md"
TEMPLATE = ADR_ROOT / "template.md"

# The reference is mirrored across the two trees under the section-parity rule, which leaves fenced
# blocks free -- and the templates are entirely fenced. Pinning the template against both copies is
# what stops the shared tree's version drifting where that rule cannot see it.
SPECS = (
    REPO / ".claude" / "references" / "architecture" / "architecture-decision-records.md",
    REPO / ".agents" / "references" / "architecture" / "architecture-decision-records.md",
)

# Both match docs/adr/**/*.md and neither is a decision. Matched at the root only: a topic
# directory is free to hold an ADR slugged `template` or `readme`.
SCAFFOLD = frozenset({"README.md", "template.md"})

REQUIRED_KEYS = ("status", "topic", "created", "updated", "deciders")

# A replaced decision is edited in place and a dead one is deleted, so neither end of a supersede
# link can exist. A file carrying one is a tombstone that should have been removed.
BANNED_KEYS = ("supersedes", "superseded-by")

LIVE_STATUSES = frozenset({"proposed", "accepted"})

# Sections the living-document model retired. Named so a copy-paste from an older ADR fails loudly
# instead of quietly reintroducing an append-only log or a second place to look for alternatives.
# Compared casefolded, so `## amendment log` does not slip past the exact spelling.
RETIRED_SECTIONS = frozenset(
    {
        "amendment log",
        "considered options",
        "implementation notes",
        "related decisions",
    }
)

# The floor both sectioned templates share. The Standard template adds Decision Drivers, Rationale,
# and Enforcement; requiring those too would make the Lightweight template illegal. The Y-Statement
# format is a single sentence with no `##` headings at all, so this applies only once a record has
# chosen to have sections.
REQUIRED_SECTIONS = ("Context", "Decision", "Consequences", "Ruled Out", "Reversal Conditions")

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Ruled-out ideas are named, never numbered: a table row has no anchor, so "Option 2" stops
# resolving the moment the table is reordered. A letter is a label too, so `Option B` is the same
# defect -- matched case-sensitively there, because `options I saw` is ordinary prose.
NUMBERED_OPTION = re.compile(r"\boptions?\s+#?\d", re.IGNORECASE)
LETTERED_OPTION = re.compile(r"\bOption [A-Z]\b")

MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

# `status: accepted  # in force` is the form the reference's own frontmatter example ships. Without
# this the documented example fails the tests that read it.
YAML_COMMENT = re.compile(r"\s+#.*$")


def adr_paths() -> list[Path]:
    """Every live decision record. Scaffolding, directories, and symlinks are not records.

    `is_file()` follows a symlink, so the containment check is what keeps a link planted under
    `docs/adr/` from pulling an arbitrary file on disk into the assertion messages.
    """
    out = []
    for path in sorted(ADR_ROOT.rglob("*.md")):
        if path.parent == ADR_ROOT and path.name in SCAFFOLD:
            continue
        if not path.is_file():
            continue
        if path.resolve().parent.is_relative_to(ADR_ROOT.resolve()):
            out.append(path)
    return out


def frontmatter(text: str) -> dict[str, str]:
    """The YAML block between the leading `---` fences, as raw string values.

    Deliberately not a YAML parser: the frontmatter is five flat scalar keys, and this repo ships
    no runtime pip dependencies, so there is no yaml module to reach for. It is strict about the
    closing fence, though -- without that check an unterminated block reads the whole document as
    frontmatter, which both invents keys from body lines and reports a malformed record as valid.
    Returns `{}` when there is no well-formed block, which every caller reports by name.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return fields
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = YAML_COMMENT.sub("", value.strip())
    return {}


def sections(text: str) -> dict[str, list[str]]:
    """Body lines grouped under their `##` heading, fenced blocks included as body."""
    out: dict[str, list[str]] = {}
    current = ""
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            out.setdefault(current, [])
        elif current:
            out[current].append(line)
    return out


def parse_date(value: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


class AdrSetTests(unittest.TestCase):
    def test_the_decision_log_is_not_empty(self):
        """Deleting a dead ADR is legal here, so every other test in this file loops over a set
        that can legitimately shrink. If it reaches zero the loops stop asserting and the suite
        goes green on a decision log that no longer exists."""
        self.assertTrue(
            adr_paths(),
            f"{ADR_ROOT.relative_to(REPO)} yielded no ADRs; every check in this file is vacuous",
        )

    def test_there_is_no_archive_directory(self):
        """A retired record is deleted, not filed away. A directory named for the archive this
        model rejects is the shape a reintroduction would take."""
        for path in sorted(ADR_ROOT.rglob("*")):
            if path.is_dir():
                with self.subTest(path=path.relative_to(REPO)):
                    self.assertNotIn(
                        path.name.lower(),
                        {"archive", "archived", "superseded", "deprecated"},
                        f"{path.relative_to(REPO)}: dead ADRs are deleted, not archived",
                    )


class AdrFrontmatterTests(unittest.TestCase):
    def test_every_adr_declares_the_required_frontmatter(self):
        for path in adr_paths():
            with self.subTest(path=path.relative_to(REPO)):
                rel = path.relative_to(REPO)
                fields = frontmatter(path.read_text(encoding="utf-8"))
                self.assertTrue(
                    fields, f"{rel}: no frontmatter block, or its closing `---` is missing"
                )
                missing = [key for key in REQUIRED_KEYS if key not in fields]
                self.assertFalse(missing, f"{rel}: frontmatter missing {missing}")
                self.assertIn(
                    fields["status"],
                    LIVE_STATUSES,
                    f"{rel}: status {fields['status']!r} not in {sorted(LIVE_STATUSES)}",
                )

    def test_no_adr_carries_a_supersede_link(self):
        for path in adr_paths():
            with self.subTest(path=path.relative_to(REPO)):
                fields = frontmatter(path.read_text(encoding="utf-8"))
                present = [key for key in BANNED_KEYS if key in fields]
                self.assertFalse(
                    present,
                    f"{path.relative_to(REPO)}: retired frontmatter {present}; a superseded ADR is "
                    "deleted, not linked",
                )

    def test_updated_is_never_before_created(self):
        for path in adr_paths():
            with self.subTest(path=path.relative_to(REPO)):
                rel = path.relative_to(REPO)
                fields = frontmatter(path.read_text(encoding="utf-8"))
                dates = {}
                for key in ("created", "updated"):
                    raw = fields.get(key, "")
                    self.assertRegex(raw, ISO_DATE, f"{rel}: {key} is not YYYY-MM-DD")
                    parsed = parse_date(raw)
                    self.assertIsNotNone(parsed, f"{rel}: {key} {raw!r} is not a real date")
                    dates[key] = parsed
                self.assertGreaterEqual(
                    dates["updated"], dates["created"], f"{rel}: updated precedes created"
                )

    def test_topic_matches_the_containing_directory(self):
        for path in adr_paths():
            with self.subTest(path=path.relative_to(REPO)):
                fields = frontmatter(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    fields.get("topic", ""),
                    path.parent.name,
                    f"{path.relative_to(REPO)}: topic field and directory disagree",
                )

    def test_every_topic_is_registered_in_the_readme(self):
        """`docs/adr/README.md` is the topic list. Adding a directory and forgetting the row is
        the one shared-state edit the naming scheme still requires."""
        listed = README.read_text(encoding="utf-8")
        for path in adr_paths():
            with self.subTest(path=path.relative_to(REPO)):
                topic = path.parent.name
                self.assertIn(
                    f"`{topic}/`",
                    listed,
                    f"{path.relative_to(REPO)}: topic {topic!r} has no row in "
                    f"{README.relative_to(REPO)}",
                )


class AdrSectionTests(unittest.TestCase):
    def test_no_adr_leaves_a_fence_unterminated(self):
        """An unclosed ``` swallows every heading after it, so the section checks below would
        report a clean document. They cannot distinguish that from a fence-free tail; this can."""
        for path in adr_paths():
            with self.subTest(path=path.relative_to(REPO)):
                self.assertFalse(
                    has_unterminated_fence(path.read_text(encoding="utf-8")),
                    f"{path.relative_to(REPO)}: a fenced block is never closed, which hides every "
                    "heading after it from the section checks",
                )

    def test_no_adr_keeps_a_retired_section(self):
        for path in adr_paths():
            with self.subTest(path=path.relative_to(REPO)):
                names = {
                    name.casefold() for name in heading_texts(path.read_text(encoding="utf-8"))
                }
                stale = sorted(names & RETIRED_SECTIONS)
                self.assertFalse(stale, f"{path.relative_to(REPO)}: retired section(s) {stale}")

    def test_every_sectioned_adr_carries_the_required_sections(self):
        """Retiring `## Amendment Log` only works if reversals have somewhere else to land.
        `## Ruled Out` is that place, and every consumer now points at it."""
        for path in adr_paths():
            with self.subTest(path=path.relative_to(REPO)):
                names = heading_texts(path.read_text(encoding="utf-8"))
                if not names:
                    continue  # Y-Statement format: one sentence, no sections by design.
                missing = [name for name in REQUIRED_SECTIONS if name not in names]
                self.assertFalse(missing, f"{path.relative_to(REPO)}: missing section(s) {missing}")

    def test_related_links_resolve(self):
        """Deleting a dead ADR orphans every inbound link, which is the one failure the
        delete-don't-archive decision actively creates."""
        for path in adr_paths():
            related = sections(path.read_text(encoding="utf-8")).get("Related", [])
            for target in MARKDOWN_LINK.findall("\n".join(related)):
                if target.startswith(("http://", "https://", "#")):
                    continue
                with self.subTest(path=path.relative_to(REPO), target=target):
                    resolved = (path.parent / target.split("#", 1)[0]).resolve()
                    self.assertTrue(
                        resolved.exists(),
                        f"{path.relative_to(REPO)}: `## Related` link {target!r} resolves nowhere",
                    )

    def test_no_adr_cites_an_option_by_number_or_letter(self):
        for path in adr_paths():
            hits = [
                f"line {number}: {line.strip()}"
                for number, line in lines_outside_fences(path.read_text(encoding="utf-8"))
                if NUMBERED_OPTION.search(line) or LETTERED_OPTION.search(line)
            ]
            with self.subTest(path=path.relative_to(REPO)):
                self.assertFalse(
                    hits,
                    f"{path.relative_to(REPO)}: cites an option by label. `## Ruled Out` rows have "
                    f"no anchor, so the label resolves nowhere -- name the idea instead: {hits}",
                )


class AdrTemplateTests(unittest.TestCase):
    def test_template_matches_the_standard_template_in_every_spec_copy(self):
        """docs/adr/README.md calls template.md a verbatim copy of the spec's Standard ADR block.

        This is what makes that claim true, in both trees: edit one and the others have to follow.
        """
        wanted = TEMPLATE.read_text(encoding="utf-8")
        for spec in SPECS:
            with self.subTest(spec=spec.relative_to(REPO)):
                rel = spec.relative_to(REPO)
                text = spec.read_text(encoding="utf-8")
                self.assertIn("### Standard ADR", text, f"{rel}: no Standard ADR section")
                after = text.split("### Standard ADR", 1)[1]
                block = re.search(r"```markdown\n(.*?)```", after, re.DOTALL)
                self.assertIsNotNone(block, f"{rel}: Standard ADR section has no ```markdown block")
                assert block is not None
                found = block.group(1)
                self.assertTrue(
                    found.startswith("---\nstatus: proposed"),
                    f"{rel}: the first ```markdown block after `### Standard ADR` is not the "
                    f"template -- check the fence tag and what sits between them. Got: {found[:40]!r}",
                )
                self.assertEqual(
                    wanted,
                    found,
                    f"docs/adr/template.md has drifted from the Standard ADR template in {rel}",
                )


class ParserTests(unittest.TestCase):
    """The two hand-rolled readers above carry every other test in this file. Their edge cases are
    asserted here rather than left to whatever the four live ADRs happen to exercise."""

    def test_frontmatter_reads_a_well_formed_block(self):
        self.assertEqual(
            frontmatter("---\nstatus: accepted\ntopic: workflow\n---\n# T\n"),
            {"status": "accepted", "topic": "workflow"},
        )

    def test_frontmatter_rejects_an_unterminated_block(self):
        self.assertEqual(frontmatter("---\nstatus: accepted\n\n# T\nsuperseded-by: x\n"), {})

    def test_frontmatter_rejects_a_document_with_no_block(self):
        self.assertEqual(frontmatter("\n---\nstatus: accepted\n---\n"), {})

    def test_frontmatter_strips_the_inline_comments_the_spec_example_ships(self):
        self.assertEqual(
            frontmatter(
                "---\nstatus: accepted  # proposed | accepted\ncreated: 2026-03-01  # set\n---\n"
            ),
            {"status": "accepted", "created": "2026-03-01"},
        )

    def test_unterminated_fence_is_detected(self):
        self.assertTrue(has_unterminated_fence("```\nx\n\n## Amendment Log\n"))
        self.assertFalse(has_unterminated_fence("```\nx\n```\n\n## Amendment Log\n"))

    def test_headings_inside_a_closed_fence_are_not_sections(self):
        self.assertEqual(heading_texts("## Real\n\n```markdown\n## Example\n```\n"), {"Real"})

    def test_option_labels_are_matched_in_every_form_that_fails_to_anchor(self):
        for text in ("Option 2", "option 2", "Options  3 and 4", "Option #2", "Option B"):
            with self.subTest(text=text):
                self.assertTrue(NUMBERED_OPTION.search(text) or LETTERED_OPTION.search(text), text)
        for text in ("options I weighed", "optional 2 steps", "the option to skip"):
            with self.subTest(text=text):
                self.assertFalse(NUMBERED_OPTION.search(text) or LETTERED_OPTION.search(text), text)


if __name__ == "__main__":
    unittest.main()
