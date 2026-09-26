"""Behaviour tests for the memory hygiene check.

`memory.py` reads Claude Code auto-memory, which holds private notes about the user's projects. The
suite never points it at the real stores: `run_script` refuses to launch without `--memory-dir`, so a
test cannot fall back to `~/.claude` by forgetting a flag.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / ".claude" / "skills" / "skill-audit" / "scripts" / "memory.py"


def memory_file(
    name: str, body: str = "Rule.", kind: str = "feedback", description: str = "d"
) -> str:
    """Return a memory file's text with standard frontmatter."""
    return (
        f"---\nname: {name}\ndescription: {description}\nmetadata:\n  type: {kind}\n---\n\n{body}\n"
    )


class _StoreMixin(unittest.TestCase):
    def store(self, files: dict, index=None) -> Path:
        """Write a memory dir from {filename: text}; index=None builds one linking every file."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "memory"
        root.mkdir()
        for filename, text in files.items():
            (root / filename).write_text(text, encoding="utf-8")
        if index is None:
            index = "# Memory Index\n" + "".join(f"- [{f}]({f}) - hook\n" for f in sorted(files))
        if index is not False:
            (root / "MEMORY.md").write_text(index, encoding="utf-8")
        return root

    def run_script(self, *args: str, expect: int = 0) -> "subprocess.CompletedProcess[str]":
        """Run memory.py, requiring an explicit store so the real ~/.claude is never read."""
        self.assertIn("--memory-dir", args, "every run must name its memory dir explicitly")
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, expect, proc.stdout + proc.stderr)
        return proc

    def findings(self, *dirs: Path, expect: int = 0):
        """Run in JSON mode and return the findings list."""
        args = []
        for d in dirs:
            args += ["--memory-dir", str(d)]
        proc = self.run_script(*args, "--json", expect=expect)
        return json.loads(proc.stdout)["findings"]


class MemoryCheckTests(_StoreMixin):
    def test_clean_store_passes(self):
        root = self.store(
            {
                "feedback_a.md": memory_file("feedback-a", "See [[feedback-b]]."),
                "feedback_b.md": memory_file("feedback-b"),
            }
        )
        self.assertEqual(self.findings(root), [])

    def test_index_entry_pointing_at_a_missing_file_fails(self):
        root = self.store(
            {"feedback_a.md": memory_file("feedback-a")},
            index="- [a](feedback_a.md)\n- [gone](feedback_gone.md)\n",
        )
        found = self.findings(root, expect=1)
        self.assertIn(
            ("FAIL", "index-dangling", "feedback_gone.md"),
            {(f["level"], f["check"], f["file"]) for f in found},
        )

    def test_file_missing_from_the_index_fails(self):
        root = self.store(
            {
                "feedback_a.md": memory_file("feedback-a"),
                "feedback_b.md": memory_file("feedback-b"),
            },
            index="- [a](feedback_a.md)\n",
        )
        found = self.findings(root, expect=1)
        self.assertIn(
            ("FAIL", "unindexed", "feedback_b.md"),
            {(f["level"], f["check"], f["file"]) for f in found},
        )

    def test_store_without_an_index_fails(self):
        root = self.store({"feedback_a.md": memory_file("feedback-a")}, index=False)
        found = self.findings(root, expect=1)
        self.assertIn("index-missing", {f["check"] for f in found})

    def test_wikilink_to_an_unknown_name_fails(self):
        root = self.store({"feedback_a.md": memory_file("feedback-a", "See [[no-such-memory]].")})
        found = self.findings(root, expect=1)
        self.assertEqual([(f["check"], f["file"]) for f in found], [("wikilink", "feedback_a.md")])
        self.assertIn("no-such-memory", found[0]["detail"])

    def test_wikilink_matches_names_across_underscore_and_hyphen(self):
        root = self.store(
            {
                "feedback_a.md": memory_file("feedback-a", "See [[feedback_b]]."),
                "feedback_b.md": memory_file("feedback-b"),
            }
        )
        self.assertEqual(self.findings(root), [])

    def test_double_brackets_inside_code_are_not_wikilinks(self):
        body = (
            "TOML tables like `[[hooks.PreToolUse]]` and bash tests like `[[ -n $x ]]`.\n\n"
            "```toml\n[[package]]\nname = 'x'\n```\n"
        )
        root = self.store({"feedback_a.md": memory_file("feedback-a", body)})
        self.assertEqual(self.findings(root), [])

    def test_wikilink_may_name_a_file_stem_instead_of_its_name_field(self):
        root = self.store(
            {
                "feedback_a.md": memory_file("feedback-a", "See [[reference_long_title]]."),
                "reference_long_title.md": memory_file("short-name", kind="reference"),
            }
        )
        self.assertEqual(self.findings(root), [])

    def test_index_over_200_lines_fails_because_the_harness_truncates_it(self):
        files = {"feedback_a.md": memory_file("feedback-a")}
        index = "- [a](feedback_a.md)\n" + "".join(f"note {i}\n" for i in range(200))
        root = self.store(files, index=index)
        found = self.findings(root, expect=1)
        self.assertIn("index-too-long", {f["check"] for f in found})

    def test_long_index_line_only_warns(self):
        long_hook = "x" * 200
        root = self.store(
            {"feedback_a.md": memory_file("feedback-a")},
            index=f"- [a](feedback_a.md) - {long_hook}\n",
        )
        found = self.findings(root, expect=0)
        self.assertEqual([(f["level"], f["check"]) for f in found], [("WARN", "index-line-long")])

    def test_missing_frontmatter_field_fails(self):
        root = self.store(
            {"feedback_a.md": "---\nname: feedback-a\n---\n\nNo description or type.\n"}
        )
        found = self.findings(root, expect=1)
        details = " ".join(f["detail"] for f in found if f["check"] == "frontmatter")
        self.assertIn("description", details)
        self.assertIn("type", details)

    def test_legacy_top_level_type_is_accepted(self):
        text = "---\nname: feedback-a\ndescription: d\ntype: feedback\n---\n\nRule.\n"
        root = self.store({"feedback_a.md": text})
        self.assertEqual(self.findings(root), [])

    def test_symlinked_stores_are_checked_once(self):
        root = self.store({"feedback_a.md": memory_file("feedback-a", "See [[missing]].")})
        link = root.parent / "alias"
        os.symlink(root, link)
        found = self.findings(root, link, expect=1)
        self.assertEqual(len(found), 1)

    def test_text_report_names_the_check_and_file(self):
        root = self.store({"feedback_a.md": memory_file("feedback-a", "See [[missing]].")})
        proc = self.run_script("--memory-dir", str(root), expect=1)
        self.assertIn("FAIL", proc.stdout)
        self.assertIn("wikilink", proc.stdout)
        self.assertIn("feedback_a.md", proc.stdout)

    def test_missing_store_is_an_error(self):
        proc = self.run_script("--memory-dir", "/nonexistent/memory-store", expect=2)
        self.assertIn("not a directory", proc.stderr)


if __name__ == "__main__":
    unittest.main()
