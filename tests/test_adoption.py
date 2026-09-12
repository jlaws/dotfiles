"""Behaviour tests for the KB adoption report.

`adoption.py` reads Claude Code transcripts, which are the most sensitive thing on this machine. It
is manual-trigger only and structurally unreachable from `make verify`: no Makefile target calls it,
`audit.py` never imports it, and every test here runs against synthetic JSONL in a temp directory.
Nothing in this file touches `~/.claude/projects/`, and `test_never_reads_the_real_transcript_root`
pins that.

The privacy contract is counts only: no transcript text, no session ID, and no path out of a user's
project may appear in the output.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / ".claude" / "skills" / "skill-audit" / "scripts" / "adoption.py"

SECRET = "ThisIsVerbatimTranscriptTextAndMustNeverBePrinted"


def load_adoption():
    """Import the adoption script by path, since its directory is not a package."""
    spec = importlib.util.spec_from_file_location("kb_adoption", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def iso(days_ago: float) -> str:
    """Return an ISO-8601 UTC timestamp that many days in the past."""
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


class _TmpMixin(unittest.TestCase):
    def tmpdir(self) -> Path:
        """Return a temp directory removed when the test ends."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def write_transcript(self, root: Path, project: str, name: str, records) -> Path:
        """Write one synthetic JSONL transcript and return its path.

        Records default to `type: user`, so a test only spells the type out when that is the thing
        under test."""
        directory = root / project
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (name + ".jsonl")
        path.write_text("".join(json.dumps(dict({"type": "user"}, **r)) + "\n" for r in records))
        return path

    def run_script(self, *args: str) -> "subprocess.CompletedProcess[str]":
        """Run adoption.py as a subprocess and require exit 0."""
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(REPO), *args],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "", proc.stderr)
        return proc


class AdoptionCountingTests(_TmpMixin):
    def rows(self, root: Path, *args: str):
        """Run in JSON mode against a transcript root and return rows keyed by asset name."""
        proc = self.run_script("--transcripts", str(root), "--json", *args)
        return {row["name"]: row for row in json.loads(proc.stdout)["assets"]}

    def test_counts_a_mention_once_per_record(self):
        root = self.tmpdir()
        self.write_transcript(
            root,
            "project-a",
            "s1",
            [
                {"sessionId": "s1", "timestamp": iso(1), "text": "run the skill-audit skill"},
                {"sessionId": "s1", "timestamp": iso(1), "text": "skill-audit again, skill-audit"},
                {"sessionId": "s1", "timestamp": iso(1), "text": "nothing relevant here"},
            ],
        )
        rows = self.rows(root)
        self.assertEqual(rows["skill-audit"]["mentions"], 2)

    def test_an_unmentioned_asset_reports_no_evidence_not_unused(self):
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [{"sessionId": "s1", "timestamp": iso(1), "text": "hi"}])
        row = self.rows(root)["skill-audit"]
        self.assertEqual(row["mentions"], 0)
        self.assertEqual(row["status"], "no evidence")
        self.assertIsNone(row["last_seen"])

    def test_a_substring_of_a_longer_name_is_not_a_mention(self):
        """`cmd-j-plan` is the shared skill, not the `j-plan` command. Counting it would inflate
        every command by the number of times its own shared skill was named."""
        root = self.tmpdir()
        self.write_transcript(
            root, "project-a", "s1",
            [{"sessionId": "s1", "timestamp": iso(1), "text": "see cmd-j-plan for details"}],
        )
        self.assertEqual(self.rows(root)["j-plan"]["mentions"], 0)

    def test_only_conversation_turns_are_counted(self):
        """The harness injects hook output and its own catalogue of every agent and skill. Counting
        those records reports the catalogue instead of the session: measured over the real
        transcripts, it marked all 79 assets active and left nothing silent."""
        root = self.tmpdir()
        boilerplate = [
            {
                "type": "attachment",
                "timestamp": iso(1),
                "attachment": {"hookEvent": "SessionStart", "content": "check writing-plans"},
            },
            {"type": "user", "timestamp": iso(1), "isMeta": True, "text": "writing-plans"},
            {"type": "ai-title", "timestamp": iso(1), "text": "writing-plans"},
            {"type": "user", "timestamp": iso(1), "attachment": {"x": 1}, "text": "writing-plans"},
        ]
        self.write_transcript(root, "project-a", "s1", boilerplate)
        self.assertEqual(self.rows(root)["writing-plans"]["mentions"], 0)

        self.write_transcript(
            root, "project-a", "s2",
            [*boilerplate, {"type": "assistant", "timestamp": iso(1), "text": "use writing-plans"}],
        )
        self.assertEqual(self.rows(root)["writing-plans"]["mentions"], 1)

    def test_fixture_directories_are_skipped(self):
        root = self.tmpdir()
        self.write_transcript(
            root, "-private-var-folders-T-ralph-e2e-abc-ralph-test", "s1",
            [{"sessionId": "s1", "timestamp": iso(1), "text": "skill-audit skill-audit"}],
        )
        self.assertEqual(self.rows(root)["skill-audit"]["mentions"], 0)

    def test_the_since_window_excludes_older_records(self):
        root = self.tmpdir()
        self.write_transcript(
            root, "project-a", "s1",
            [
                {"sessionId": "s1", "timestamp": iso(3), "text": "skill-audit"},
                {"sessionId": "s1", "timestamp": iso(90), "text": "skill-audit"},
            ],
        )
        self.assertEqual(self.rows(root, "--since", "30")["skill-audit"]["mentions"], 1)
        self.assertEqual(self.rows(root, "--all")["skill-audit"]["mentions"], 2)

    def test_a_recent_mention_is_active_and_an_old_one_is_cold(self):
        root = self.tmpdir()
        self.write_transcript(
            root, "project-a", "s1", [{"sessionId": "s1", "timestamp": iso(2), "text": "skill-audit"}]
        )
        self.write_transcript(
            root, "project-a", "s2", [{"sessionId": "s2", "timestamp": iso(20), "text": "writing-plans"}]
        )
        rows = self.rows(root)
        self.assertEqual(rows["skill-audit"]["status"], "active")
        self.assertEqual(rows["writing-plans"]["status"], "cold")

    def test_a_record_with_no_timestamp_is_counted_only_with_all(self):
        """A window cannot include a record it cannot date. Dropping it silently would understate
        the count without saying so."""
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [{"sessionId": "s1", "text": "skill-audit"}])
        self.assertEqual(self.rows(root)["skill-audit"]["mentions"], 0)
        self.assertEqual(self.rows(root, "--all")["skill-audit"]["mentions"], 1)


class AdoptionPrivacyTests(_TmpMixin):
    def test_output_carries_no_transcript_text_session_id_or_project_path(self):
        root = self.tmpdir()
        self.write_transcript(
            root, "-Users-someone-Workspace-private-thing", "sess-abc-123",
            [{"sessionId": "sess-abc-123", "timestamp": iso(1), "text": SECRET + " skill-audit"}],
        )
        for args in ((), ("--json",)):
            with self.subTest(args=args):
                out = self.run_script("--transcripts", str(root), *args).stdout
                self.assertNotIn(SECRET, out)
                self.assertNotIn("sess-abc-123", out)
                self.assertNotIn("someone", out)
                self.assertNotIn("private-thing", out)

    def test_never_reads_the_real_transcript_root(self):
        """`make verify` must not touch the real transcript root. Its default is only ever reached
        by a human typing the command, so every test here passes `--transcripts` instead."""
        # Split so the needle does not match the line that defines it.
        needle = ".claude/" + "projects"
        body = Path(__file__).read_text().split('"""', 2)[2]
        hits = [n for n, line in enumerate(body.splitlines(), 1) if needle in line]
        self.assertEqual(hits, [], "test body reaches the real transcript root on line(s) " + str(hits))


class AdoptionDegradedModeTests(_TmpMixin):
    def test_a_missing_transcript_root_says_so_and_exits_zero(self):
        missing = self.tmpdir() / "not-here"
        out = self.run_script("--transcripts", str(missing)).stdout
        self.assertIn("no transcripts found", out)

    def test_an_unparseable_line_does_not_abort_the_scan(self):
        root = self.tmpdir()
        path = self.write_transcript(
            root, "project-a", "s1", [{"sessionId": "s1", "timestamp": iso(1), "text": "skill-audit"}]
        )
        # The truncated line must name an asset, or the scan skips it before it ever parses JSON
        # and the test proves nothing. A half-written last line is normal in a live transcript.
        path.write_text('{"text": "skill-audit"\n' + path.read_text())
        proc = self.run_script("--transcripts", str(root), "--json")
        rows = {row["name"]: row for row in json.loads(proc.stdout)["assets"]}
        self.assertEqual(rows["skill-audit"]["mentions"], 1)


class AdoptionReportTests(_TmpMixin):
    def test_the_text_report_leads_with_its_limits(self):
        root = self.tmpdir()
        self.write_transcript(
            root, "project-a", "s1", [{"sessionId": "s1", "timestamp": iso(1), "text": "skill-audit"}]
        )
        out = self.run_script("--transcripts", str(root)).stdout
        head = out.split("| Asset")[0]
        self.assertIn("lower bound", head)
        self.assertIn("no evidence", head)
        for limit in ("conversation turns", "read rather than invoked", "pruned", "Fixture"):
            with self.subTest(limit=limit):
                self.assertIn(limit, head)

    def test_every_claude_skill_agent_and_command_gets_a_row(self):
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [{"sessionId": "s1", "timestamp": iso(1), "text": "x"}])
        proc = self.run_script("--transcripts", str(root), "--json")
        rows = {(row["kind"], row["name"]) for row in json.loads(proc.stdout)["assets"]}
        for path in (REPO / ".claude" / "skills").glob("*/SKILL.md"):
            self.assertIn(("skill", path.parent.name), rows)
        for path in (REPO / ".claude" / "agents").glob("*.md"):
            self.assertIn(("agent", path.stem), rows)
        for path in (REPO / ".claude" / "commands").glob("*.md"):
            self.assertIn(("command", path.stem), rows)


if __name__ == "__main__":
    unittest.main()
