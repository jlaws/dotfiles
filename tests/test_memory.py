"""Behaviour tests for the memory hygiene check.

`memory.py` reads Claude Code auto-memory, which holds private notes about the user's projects. The
suite never points it at the real stores: `run_script` refuses to launch without `--memory-dir`
unless the subprocess gets a throwaway `HOME`, so a test cannot fall back to `~/.claude` by
forgetting a flag.
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
    def tmpdir(self) -> Path:
        """Return a fresh temporary directory removed at test end."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def store(self, files: dict, index=None, root=None) -> Path:
        """Write a memory dir from {filename: text}; index=None builds one linking every file."""
        root = root or self.tmpdir() / "memory"
        root.mkdir(parents=True)
        for filename, text in files.items():
            if isinstance(text, bytes):
                (root / filename).write_bytes(text)
            else:
                (root / filename).write_text(text, encoding="utf-8")
        if index is None:
            index = "# Memory Index\n" + "".join(f"- [{f}]({f}) - hook\n" for f in sorted(files))
        if index is not False:
            (root / "MEMORY.md").write_text(index, encoding="utf-8")
        return root

    def run_script(
        self, *args: str, expect: int = 0, home: Path | None = None
    ) -> "subprocess.CompletedProcess[str]":
        """Run memory.py; require an explicit store or a throwaway HOME so ~/.claude is never read."""
        if home is None:
            self.assertIn("--memory-dir", args, "every run must name its memory dir explicitly")
        env = dict(os.environ)
        if home is not None:
            env["HOME"] = str(home)
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
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

    def checks(self, found) -> set:
        """Return the (level, check, file) triples of a findings list."""
        return {(f["level"], f["check"], f["file"]) for f in found}


class MemoryCheckTests(_StoreMixin):
    def test_clean_store_passes(self):
        root = self.store(
            {
                "feedback_a.md": memory_file("feedback-a", "See [[feedback-b]]."),
                "feedback_b.md": memory_file("feedback-b"),
            }
        )
        self.assertEqual(self.findings(root), [])

    def test_empty_store_passes(self):
        root = self.store({}, index=False)
        self.assertEqual(self.findings(root), [])

    def test_index_entry_pointing_at_a_missing_file_fails(self):
        root = self.store(
            {"feedback_a.md": memory_file("feedback-a")},
            index="- [a](feedback_a.md)\n- [gone](feedback_gone.md)\n",
        )
        found = self.findings(root, expect=1)
        self.assertIn(("FAIL", "index-dangling", "feedback_gone.md"), self.checks(found))

    def test_file_missing_from_the_index_fails(self):
        root = self.store(
            {
                "feedback_a.md": memory_file("feedback-a"),
                "feedback_b.md": memory_file("feedback-b"),
            },
            index="- [a](feedback_a.md)\n",
        )
        found = self.findings(root, expect=1)
        self.assertIn(("FAIL", "unindexed", "feedback_b.md"), self.checks(found))

    def test_index_links_with_dot_slash_and_anchor_resolve(self):
        root = self.store(
            {
                "feedback_a.md": memory_file("feedback-a"),
                "feedback_b.md": memory_file("feedback-b"),
            },
            index="- [a](./feedback_a.md)\n- [b](feedback_b.md#why)\n",
        )
        self.assertEqual(self.findings(root), [])

    def test_index_link_to_a_url_is_ignored(self):
        root = self.store(
            {"feedback_a.md": memory_file("feedback-a")},
            index="- [a](feedback_a.md) - see [spec](https://example.com/SPEC.md)\n",
        )
        self.assertEqual(self.findings(root), [])

    def test_index_link_escaping_the_store_is_dangling(self):
        base = self.tmpdir()
        (base / "outside.md").write_text("secret", encoding="utf-8")
        root = self.store(
            {"feedback_a.md": memory_file("feedback-a")},
            index="- [a](feedback_a.md)\n- [o](../outside.md)\n",
            root=base / "memory",
        )
        found = self.findings(root, expect=1)
        self.assertEqual([f["check"] for f in found], ["index-dangling"])

    def test_store_without_an_index_fails(self):
        root = self.store({"feedback_a.md": memory_file("feedback-a")}, index=False)
        found = self.findings(root, expect=1)
        self.assertEqual([f["check"] for f in found], ["index-missing"])

    def test_wikilink_to_an_unknown_name_fails(self):
        root = self.store({"feedback_a.md": memory_file("feedback-a", "See [[no-such-memory]].")})
        found = self.findings(root, expect=1)
        self.assertEqual([(f["check"], f["file"]) for f in found], [("wikilink", "feedback_a.md")])
        self.assertIn("[[no-such-memory]]", found[0]["detail"])

    def test_wikilink_that_is_not_a_slug_is_reported_by_length_only(self):
        secret = "the db password is hunter2"
        root = self.store({"feedback_a.md": memory_file("feedback-a", f"See [[{secret}]].")})
        found = self.findings(root, expect=1)
        self.assertEqual(found[0]["check"], "wikilink")
        self.assertNotIn("hunter2", found[0]["detail"])
        self.assertIn(f"<{len(secret)} chars>", found[0]["detail"])

    def test_wikilink_matches_names_across_underscore_and_hyphen(self):
        root = self.store(
            {
                "feedback_a.md": memory_file("feedback-a", "See [[feedback_b]]."),
                "feedback_b.md": memory_file("feedback-b"),
            }
        )
        self.assertEqual(self.findings(root), [])

    def test_wikilink_resolves_by_name_field_not_only_file_stem(self):
        root = self.store(
            {
                "feedback_a.md": memory_file("feedback-a", "See [[short-name]]."),
                "reference_long_title.md": memory_file("short-name", kind="reference"),
            }
        )
        self.assertEqual(self.findings(root), [])

    def test_wikilink_alias_and_heading_resolve_by_slug(self):
        body = "See [[feedback-b|the b rule]] and [[feedback-b#why]]."
        root = self.store(
            {
                "feedback_a.md": memory_file("feedback-a", body),
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

    def test_index_at_200_lines_passes_and_201_fails(self):
        files = {"feedback_a.md": memory_file("feedback-a")}
        ok = "- [a](feedback_a.md)\n" + "".join(f"note {i}\n" for i in range(199))
        self.assertEqual(self.findings(self.store(files, index=ok)), [])
        over = ok + "note 199\n"
        found = self.findings(self.store(files, index=over), expect=1)
        self.assertEqual([f["check"] for f in found], ["index-too-long"])

    def test_index_over_25kb_fails_even_under_200_lines(self):
        files = {"feedback_a.md": memory_file("feedback-a")}
        index = "- [a](feedback_a.md)\n" + "".join("x" * 140 + "\n" for _ in range(190))
        self.assertLess(index.count("\n"), 200)
        self.assertGreater(len(index.encode()), 25 * 1024)
        found = self.findings(self.store(files, index=index), expect=1)
        self.assertIn("index-too-long", {f["check"] for f in found})

    def test_index_line_at_150_chars_passes_and_151_warns(self):
        files = {"feedback_a.md": memory_file("feedback-a")}
        prefix = "- [a](feedback_a.md) - "
        ok = prefix + "x" * (150 - len(prefix)) + "\n"
        self.assertEqual(self.findings(self.store(files, index=ok)), [])
        found = self.findings(self.store(files, index=prefix + "x" * (151 - len(prefix)) + "\n"))
        self.assertEqual([(f["level"], f["check"]) for f in found], [("WARN", "index-line-long")])

    def test_missing_frontmatter_field_fails(self):
        root = self.store(
            {"feedback_a.md": "---\nname: feedback-a\n---\n\nNo description or type.\n"}
        )
        found = self.findings(root, expect=1)
        details = " ".join(f["detail"] for f in found if f["check"] == "frontmatter")
        self.assertIn("description", details)
        self.assertIn("type", details)

    def test_missing_name_alone_fails(self):
        root = self.store({"feedback_a.md": "---\ndescription: d\ntype: feedback\n---\n\nRule.\n"})
        found = self.findings(root, expect=1)
        self.assertEqual(
            [(f["check"], f["detail"]) for f in found], [("frontmatter", "missing name")]
        )

    def test_quoted_empty_frontmatter_value_fails(self):
        text = '---\nname: feedback-a\ndescription: ""\ntype: feedback\n---\n\nRule.\n'
        found = self.findings(self.store({"feedback_a.md": text}), expect=1)
        self.assertEqual([f["detail"] for f in found], ["missing description"])

    def test_utf8_bom_is_tolerated(self):
        text = "﻿" + memory_file("feedback-a")
        self.assertEqual(self.findings(self.store({"feedback_a.md": text})), [])

    def test_legacy_top_level_type_is_accepted(self):
        text = "---\nname: feedback-a\ndescription: d\ntype: feedback\n---\n\nRule.\n"
        root = self.store({"feedback_a.md": text})
        self.assertEqual(self.findings(root), [])

    def test_unreadable_entries_are_reported_and_do_not_stop_the_run(self):
        base = self.tmpdir()
        good = self.store(
            {"feedback_a.md": memory_file("feedback-a", "See [[missing]].")}, root=base / "good"
        )
        bad = self.store(
            {"feedback_a.md": memory_file("feedback-a"), "binary.md": b"\xff\xfe"},
            index="- [a](feedback_a.md)\n",
            root=base / "bad",
        )
        (bad / "attic.md").mkdir()
        os.symlink(bad / "nowhere.md", bad / "dangling.md")
        found = self.findings(good, bad, expect=1)
        self.assertIn(("FAIL", "wikilink", "feedback_a.md"), self.checks(found))
        unreadable = sorted(f["file"] for f in found if f["check"] == "unreadable")
        self.assertEqual(unreadable, ["attic.md", "binary.md", "dangling.md"])

    def test_symlink_pointing_outside_the_store_is_unreadable(self):
        base = self.tmpdir()
        (base / "outside.txt").write_text(memory_file("outside"), encoding="utf-8")
        root = self.store({"feedback_a.md": memory_file("feedback-a")}, root=base / "memory")
        os.symlink(base / "outside.txt", root / "leak.md")
        found = self.findings(root, expect=1)
        self.assertIn(("FAIL", "unreadable", "leak.md"), self.checks(found))

    def test_symlinked_stores_are_checked_once(self):
        root = self.store({"feedback_a.md": memory_file("feedback-a", "See [[missing]].")})
        link = root.parent / "alias"
        os.symlink(root, link)
        found = self.findings(root, link, expect=1)
        self.assertEqual(len(found), 1)

    def test_two_stores_are_reported_separately(self):
        base = self.tmpdir()
        one = self.store(
            {"feedback_a.md": memory_file("feedback-a")}, index=False, root=base / "one"
        )
        two = self.store(
            {"feedback_b.md": memory_file("feedback-b")}, index=False, root=base / "two"
        )
        found = self.findings(one, two, expect=1)
        self.assertEqual(
            sorted(f["store"] for f in found), sorted([str(one.resolve()), str(two.resolve())])
        )

    def test_text_report_names_the_check_and_file(self):
        root = self.store({"feedback_a.md": memory_file("feedback-a", "See [[missing]].")})
        proc = self.run_script("--memory-dir", str(root), expect=1)
        rows = [line.split() for line in proc.stdout.splitlines()[:-1]]
        self.assertEqual(
            [(r[0], r[2], r[3]) for r in rows], [("FAIL", "wikilink", "feedback_a.md:")]
        )
        self.assertEqual(proc.stdout.splitlines()[-1], "1 store(s) checked: 1 FAIL, 0 WARN")

    def test_missing_stores_are_all_named_in_one_error(self):
        proc = self.run_script(
            "--memory-dir", "/nonexistent/one", "--memory-dir", "/nonexistent/two", expect=2
        )
        self.assertIn("not a directory: /nonexistent/one, /nonexistent/two", proc.stderr)


class DefaultStoreTests(_StoreMixin):
    def test_default_run_discovers_project_and_agent_stores_under_home(self):
        home = self.tmpdir()
        self.store(
            {"feedback_a.md": memory_file("feedback-a")},
            index=False,
            root=home / ".claude" / "projects" / "p" / "memory",
        )
        self.store(
            {"note.md": memory_file("note", kind="reference")},
            index=False,
            root=home / ".claude" / "agent-memory" / "code-reviewer",
        )
        proc = self.run_script("--json", expect=1, home=home)
        report = json.loads(proc.stdout)
        self.assertEqual(
            sorted(report["stores"]),
            ["~/.claude/agent-memory/code-reviewer", "~/.claude/projects/p/memory"],
        )
        self.assertEqual([f["check"] for f in report["findings"]], ["index-missing"] * 2)

    def test_store_label_needs_a_path_boundary_after_home(self):
        base = self.tmpdir()
        home = base / "jl"
        home.mkdir()
        root = self.store(
            {"feedback_a.md": memory_file("feedback-a")}, root=base / "jlaws" / "memory"
        )
        proc = self.run_script("--memory-dir", str(root), "--json", home=home)
        self.assertEqual(json.loads(proc.stdout)["stores"], [str(root.resolve())])


if __name__ == "__main__":
    unittest.main()
