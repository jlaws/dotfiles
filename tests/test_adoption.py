"""Behaviour tests for the KB adoption report.

`adoption.py` reads Claude Code transcripts, which are the most sensitive thing on this machine.
`make test` does execute the script -- roughly twenty times, as a subprocess -- so the guarantee that
matters is not "nothing runs it" but "nothing runs it against the real transcript root". That one is
enforced structurally here: `run_script` refuses to launch without `--transcripts`, so a future test
cannot reach `~/.claude/projects` by forgetting a flag. An earlier version grepped this file for the
literal path instead, which a test that simply omitted the flag sailed straight past.

Fixtures use the record shape the harness actually writes -- `message.content` blocks, tool results
carrying `toolUseResult`, and the `cwd`/`gitBranch` envelope -- because the previous fixtures used a
flat `text` key that no real transcript contains, so they could not have caught the three counting
bugs that shipped.

The privacy contract is counts only: no transcript text, no session ID, and no path out of a user's
project may appear in the output.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / ".claude" / "skills" / "skill-audit" / "scripts" / "adoption.py"

SECRET = "ThisIsVerbatimTranscriptTextAndMustNeverBePrinted"


def iso(days_ago: float) -> str:
    """Return an ISO-8601 UTC timestamp that many days in the past."""
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def turn(text, days_ago=1, kind="user", **extra):
    """Build a conversation record in the shape the harness writes."""
    record = {
        "type": kind,
        "cwd": "/Users/someone/Workspace/project",
        "gitBranch": "main",
        "message": {"role": kind, "content": [{"type": "text", "text": text}]},
    }
    if days_ago is not None:
        record["timestamp"] = iso(days_ago)
    record.update(extra)
    return record


def tool_result(text, days_ago=1, as_string=False):
    """Build a tool-output record: an ordinary `user` record carrying `toolUseResult`.

    `as_string=True` gives the record a bare-string `message.content`. Every tool result in the
    real corpus uses a `tool_result` block instead, which prose extraction drops on its own -- the
    string form is the shape only the `toolUseResult` key can catch.
    """
    record = turn(text, days_ago)
    record["toolUseResult"] = {"type": "text", "file": {"content": text}}
    record["message"]["content"] = text if as_string else [{"type": "tool_result", "content": text}]
    return record


def tool_call(path, days_ago=1):
    """Build an assistant record whose only content is a tool call naming a path."""
    record = turn("", days_ago, kind="assistant")
    record["message"]["content"] = [
        {"type": "tool_use", "name": "Read", "input": {"file_path": path}}
    ]
    return record


class _TmpMixin(unittest.TestCase):
    def tmpdir(self) -> Path:
        """Return a temp directory removed when the test ends."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def write_transcript(self, root: Path, project: str, name: str, records, nested=False) -> Path:
        """Write one synthetic JSONL transcript and return its path.

        `nested=True` places it at `<project>/<session>/subagents/<name>.jsonl`, the layout the
        harness uses for subagent transcripts and the one a depth-2 glob cannot see.
        """
        directory = root / project
        if nested:
            directory = directory / "session-uuid" / "subagents"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (name + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return path

    def run_script(self, *args: str, expect: int = 0) -> "subprocess.CompletedProcess[str]":
        """Run adoption.py as a subprocess, requiring an explicit transcript root.

        The assertion is the privacy guard: without it, omitting `--transcripts` silently falls
        back to `~/.claude/projects` and the suite reads real conversations.
        """
        self.assertIn("--transcripts", args, "every run must name its transcript root explicitly")
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(REPO), *args],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, expect, proc.stderr)
        if expect == 0:
            self.assertEqual(proc.stderr, "", proc.stderr)
        return proc

    def rows(self, root: Path, *args: str):
        """Run in JSON mode against a transcript root and return rows keyed by asset name."""
        proc = self.run_script("--transcripts", str(root), "--json", *args)
        return {row["name"]: row for row in json.loads(proc.stdout)["assets"]}

    def envelope(self, root: Path, *args: str):
        """Run in JSON mode and return the whole envelope, not just the rows."""
        return json.loads(self.run_script("--transcripts", str(root), "--json", *args).stdout)


class AdoptionCountingTests(_TmpMixin):
    def test_counts_a_mention_once_per_record(self):
        root = self.tmpdir()
        self.write_transcript(
            root,
            "project-a",
            "s1",
            [
                turn("run the skill-audit skill"),
                turn("skill-audit again, skill-audit"),
                turn("nothing relevant here"),
            ],
        )
        self.assertEqual(self.rows(root)["skill-audit"]["mentions"], 2)

    def test_an_unmentioned_asset_reports_no_evidence_not_unused(self):
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("hi")])
        row = self.rows(root)["skill-audit"]
        self.assertEqual(row["mentions"], 0)
        self.assertEqual(row["status"], "no evidence")
        self.assertIsNone(row["last_seen"])

    def test_a_substring_of_a_longer_name_is_not_a_mention(self):
        """`cmd-j-plan` is the shared skill, not the `j-plan` command."""
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("see cmd-j-plan for details")])
        self.assertEqual(self.rows(root)["j-plan"]["mentions"], 0)

    def test_subagent_transcripts_are_scanned(self):
        """Subagent transcripts sit two directories deeper. A depth-2 glob read 3% of the corpus
        on this machine -- 44 of 1408 files -- and still called it `Every transcript`."""
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "agent-1", [turn("use skill-audit")], nested=True)
        rows = self.rows(root, "--all")
        self.assertEqual(rows["skill-audit"]["mentions"], 1)
        self.assertEqual(self.envelope(root, "--all")["transcripts"], 1)

    def test_tool_output_is_not_a_mention(self):
        """Tool results are plain `user` records carrying `toolUseResult`, so a type filter keeps
        them. One `Read` of a SKILL.md would otherwise credit every asset named inside it."""
        root = self.tmpdir()
        self.write_transcript(
            root,
            "project-a",
            "s1",
            [
                tool_result("skill-audit writing-plans"),
                tool_result("skill-audit writing-plans", as_string=True),
            ],
        )
        rows = self.rows(root)
        self.assertEqual(rows["skill-audit"]["mentions"], 0)
        self.assertEqual(rows["writing-plans"]["mentions"], 0)

    def test_a_tool_call_argument_is_not_a_mention(self):
        root = self.tmpdir()
        self.write_transcript(
            root, "project-a", "s1", [tool_call(".claude/skills/skill-audit/SKILL.md")]
        )
        self.assertEqual(self.rows(root)["skill-audit"]["mentions"], 0)

    def test_the_record_envelope_is_not_a_mention(self):
        """A branch named after the work credited that asset once per turn for a whole session:
        241 increments over the real corpus, from `gitBranch` alone."""
        root = self.tmpdir()
        self.write_transcript(
            root,
            "project-a",
            "s1",
            [
                turn("unrelated prose", gitBranch="feat/skill-audit", cwd="/w/skill-audit"),
                turn("more unrelated prose", gitBranch="feat/skill-audit"),
            ],
        )
        self.assertEqual(self.rows(root)["skill-audit"]["mentions"], 0)

    def test_harness_boilerplate_is_not_counted(self):
        root = self.tmpdir()
        boilerplate = [
            {
                "type": "attachment",
                "timestamp": iso(1),
                "attachment": {"hookEvent": "SessionStart", "content": "check writing-plans"},
            },
            dict(turn("writing-plans"), isMeta=True),
            dict(turn("writing-plans"), type="ai-title"),
            dict(turn("writing-plans"), attachment={"x": 1}),
        ]
        self.write_transcript(root, "project-a", "s1", boilerplate)
        self.assertEqual(self.rows(root)["writing-plans"]["mentions"], 0)

        self.write_transcript(
            root, "project-a", "s2", [*boilerplate, turn("use writing-plans", kind="assistant")]
        )
        self.assertEqual(self.rows(root)["writing-plans"]["mentions"], 1)

    def test_fixture_directories_are_skipped(self):
        root = self.tmpdir()
        self.write_transcript(
            root,
            "-private-var-folders-T-ralph-e2e-abc-ralph-test",
            "s1",
            [turn("skill-audit skill-audit")],
        )
        self.assertEqual(self.rows(root)["skill-audit"]["mentions"], 0)

    def test_the_exclude_flag_skips_a_named_project(self):
        """`--exclude` is the documented primary invocation and nothing covered it."""
        root = self.tmpdir()
        self.write_transcript(root, "-Users-me-Workspace-dotfiles", "s1", [turn("skill-audit")])
        self.write_transcript(root, "-Users-me-Workspace-other", "s2", [turn("skill-audit")])
        self.assertEqual(self.rows(root)["skill-audit"]["mentions"], 2)
        self.assertEqual(self.rows(root, "--exclude", "dotfiles")["skill-audit"]["mentions"], 1)

    def test_exclude_also_skips_nested_subagent_transcripts(self):
        """Exclusion keys on the project directory, which is no longer the file's parent."""
        root = self.tmpdir()
        self.write_transcript(
            root, "-Users-me-Workspace-dotfiles", "agent-1", [turn("skill-audit")], nested=True
        )
        self.assertEqual(self.rows(root, "--exclude", "dotfiles")["skill-audit"]["mentions"], 0)

    def test_an_exclude_that_matches_nothing_says_so(self):
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("skill-audit")])
        envelope = self.envelope(root, "--exclude", "typo-here")
        self.assertEqual(envelope["notes"], ["--exclude typo-here matched no project directory"])
        self.assertIn("matched no project directory", self.run_script(
            "--transcripts", str(root), "--exclude", "typo-here"
        ).stdout)

    def test_the_since_window_excludes_older_records(self):
        root = self.tmpdir()
        self.write_transcript(
            root, "project-a", "s1", [turn("skill-audit", 3), turn("skill-audit", 90)]
        )
        self.assertEqual(self.rows(root, "--since", "30")["skill-audit"]["mentions"], 1)
        self.assertEqual(self.rows(root, "--all")["skill-audit"]["mentions"], 2)

    def test_the_since_window_is_honoured_at_its_boundary(self):
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("skill-audit", 9)])
        self.assertEqual(self.rows(root, "--since", "10")["skill-audit"]["mentions"], 1)
        self.assertEqual(self.rows(root, "--since", "8")["skill-audit"]["mentions"], 0)

    def test_last_seen_reports_the_newest_mention_not_the_last_read(self):
        root = self.tmpdir()
        self.write_transcript(
            root, "project-a", "s1", [turn("skill-audit", 2), turn("skill-audit", 40)]
        )
        newest = (datetime.now(timezone.utc) - timedelta(days=2)).date().isoformat()
        self.assertEqual(self.rows(root, "--all")["skill-audit"]["last_seen"], newest)

    def test_active_and_cold_follow_the_active_days_window(self):
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("skill-audit", 2)])
        self.write_transcript(root, "project-a", "s2", [turn("writing-plans", 20)])
        rows = self.rows(root)
        self.assertEqual(rows["skill-audit"]["status"], "active")
        self.assertEqual(rows["writing-plans"]["status"], "cold")
        widened = self.rows(root, "--active-days", "30")
        self.assertEqual(widened["writing-plans"]["status"], "active")

    def test_a_record_with_no_timestamp_is_counted_only_with_all(self):
        """A window cannot include a record it cannot date. Undated but counted is `cold`, never
        `active`, because recency is exactly what is unknown."""
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("skill-audit", days_ago=None)])
        self.assertEqual(self.rows(root)["skill-audit"]["mentions"], 0)
        row = self.rows(root, "--all")["skill-audit"]
        self.assertEqual(row["mentions"], 1)
        self.assertEqual(row["status"], "cold")
        self.assertIsNone(row["last_seen"])

    def test_an_unparseable_timestamp_does_not_abort_the_scan(self):
        root = self.tmpdir()
        self.write_transcript(
            root,
            "project-a",
            "s1",
            [dict(turn("skill-audit"), timestamp="not-a-real-date"), turn("skill-audit")],
        )
        self.assertEqual(self.rows(root, "--all")["skill-audit"]["mentions"], 2)

    def test_a_z_suffixed_timestamp_is_understood(self):
        """The harness writes `...Z`, and 3.9's fromisoformat rejects both that and a fraction
        width other than 3 or 6 digits. Getting this wrong zeroes every windowed count."""
        root = self.tmpdir()
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-2] + "Z"
        self.write_transcript(
            root, "project-a", "s1", [dict(turn("skill-audit"), timestamp=stamp)]
        )
        row = self.rows(root)["skill-audit"]
        self.assertEqual(row["mentions"], 1)
        self.assertEqual(row["status"], "active")


class AdoptionPrivacyTests(_TmpMixin):
    def test_output_carries_no_transcript_text_session_id_or_project_path(self):
        root = self.tmpdir()
        self.write_transcript(
            root,
            "-Users-someone-Workspace-private-thing",
            "sess-abc-123",
            [dict(turn(SECRET + " skill-audit"), sessionId="sess-abc-123")],
        )
        for args in ((), ("--json",)):
            with self.subTest(args=args):
                out = self.run_script("--transcripts", str(root), *args).stdout
                self.assertNotIn(SECRET, out)
                self.assertNotIn("sess-abc-123", out)
                self.assertNotIn("someone", out)
                self.assertNotIn("private-thing", out)

    def test_last_seen_is_a_date_not_a_timestamp(self):
        """The only clock value that reaches stdout. A full timestamp would place a user at a
        keyboard to the microsecond."""
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("skill-audit", 1)])
        seen = self.rows(root)["skill-audit"]["last_seen"]
        self.assertRegex(seen, r"^\d{4}-\d{2}-\d{2}$")

    def test_an_unreadable_transcript_is_counted_but_never_named(self):
        """The path is the project directory plus the session ID. An unguarded open raised
        PermissionError and printed both to stderr at exit 1."""
        root = self.tmpdir()
        path = self.write_transcript(
            root, "-Users-someone-Workspace-private-thing", "sess-abc-123", [turn("skill-audit")]
        )
        os.chmod(path, 0o000)
        self.addCleanup(os.chmod, path, 0o644)
        if os.access(path, os.R_OK):  # running as root
            self.skipTest("cannot make a file unreadable as this user")
        proc = self.run_script("--transcripts", str(root), "--json")
        envelope = json.loads(proc.stdout)
        self.assertEqual(envelope["unreadable"], 1)
        self.assertEqual(envelope["transcripts"], 0)
        self.assertNotIn("sess-abc-123", proc.stdout + proc.stderr)
        self.assertNotIn("private-thing", proc.stdout + proc.stderr)

    def test_a_directory_named_like_a_transcript_does_not_crash_the_scan(self):
        root = self.tmpdir()
        (root / "project-a" / "weird.jsonl").mkdir(parents=True)
        self.write_transcript(root, "project-a", "s1", [turn("skill-audit")])
        envelope = self.envelope(root, "--json")
        self.assertEqual(envelope["unreadable"], 1)
        self.assertEqual(envelope["assets"] and envelope["transcripts"], 1)

    def test_an_asset_name_cannot_break_out_of_the_report_table(self):
        """Names come from directory stems. A hostile checkout could otherwise inject rows into a
        report the skill-audit agent then reads."""
        repo = self.tmpdir()
        (repo / ".claude" / "skills" / "x | INJECTED | y").mkdir(parents=True)
        (repo / ".claude" / "skills" / "x | INJECTED | y" / "SKILL.md").write_text("x")
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("hi")])
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(repo), "--transcripts", str(root)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = proc.stdout.split("|---|---|---|---|---|")[1]
        self.assertNotIn("| INJECTED |", body)
        self.assertIn("\\| INJECTED \\|", body)


class AdoptionDegradedModeTests(_TmpMixin):
    def test_a_missing_transcript_root_says_so_and_exits_zero(self):
        missing = self.tmpdir() / "not-here"
        out = self.run_script("--transcripts", str(missing)).stdout
        self.assertIn("no transcripts found", out)

    def test_a_missing_transcript_root_still_emits_json_when_asked(self):
        """A caller parsing `--json` got plain text here and a JSONDecodeError at exit 0."""
        missing = self.tmpdir() / "not-here"
        out = self.run_script("--transcripts", str(missing), "--json").stdout
        self.assertEqual(json.loads(out)["assets"], [])

    def test_a_repo_with_no_knowledge_base_is_an_error_not_an_empty_report(self):
        """An empty table and `0 active | 0 cold | 0 no evidence` reads as a real measurement."""
        empty = self.tmpdir()
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("skill-audit")])
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(empty), "--transcripts", str(root)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("no knowledge-base assets found", proc.stdout)

    def test_an_unparseable_line_does_not_abort_the_scan(self):
        root = self.tmpdir()
        path = self.write_transcript(root, "project-a", "s1", [turn("skill-audit")])
        # The truncated line must name an asset, or the scan skips it before it ever parses JSON
        # and the test proves nothing. A half-written last line is normal in a live transcript.
        path.write_text('{"text": "skill-audit"\n' + path.read_text(), encoding="utf-8")
        self.assertEqual(self.rows(root, "--json")["skill-audit"]["mentions"], 1)

    def test_a_window_shorter_than_a_day_is_rejected(self):
        root = self.tmpdir()
        for bad in (("--since", "0"), ("--since", "-5"), ("--active-days", "-1")):
            with self.subTest(bad=bad):
                proc = subprocess.run(
                    [sys.executable, str(SCRIPT), str(REPO), "--transcripts", str(root), *bad],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(proc.returncode, 2, proc.stdout)

    def test_an_empty_exclude_is_rejected(self):
        """`"" in name` is always True, so this would have excluded every project silently."""
        root = self.tmpdir()
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(REPO), "--transcripts", str(root), "--exclude", ""],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 2, proc.stdout)


class AdoptionReportTests(_TmpMixin):
    def test_the_text_report_leads_with_its_limits(self):
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("skill-audit")])
        out = self.run_script("--transcripts", str(root)).stdout
        head = out.split("| Asset")[0]
        self.assertIn("lower bound", head)
        for limit in ("Tool calls, tool output", "record envelope", "pruned", "Fixture"):
            with self.subTest(limit=limit):
                self.assertIn(limit, head)

    def test_the_legend_does_not_call_cold_an_absence_of_evidence(self):
        """`classify` returns `no evidence` only at count 0, so every `cold` row has evidence.
        Calling them the same thing sent the orphan judgement after assets that were in use."""
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("skill-audit", 40)])
        head = self.run_script("--transcripts", str(root), "--all").stdout.split("| Asset")[0]
        self.assertIn("`cold` means it was found", head)
        self.assertNotIn("both mean no evidence was found", head)

    def test_the_totals_line_counts_each_status(self):
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("skill-audit", 1)])
        self.write_transcript(root, "project-a", "s2", [turn("writing-plans", 40)])
        out = self.run_script("--transcripts", str(root), "--all").stdout
        total = len(self.rows(root, "--all"))
        self.assertIn("1 active | 1 cold | {} no evidence".format(total - 2), out)

    def test_the_scanned_file_count_is_reported(self):
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("skill-audit")])
        self.write_transcript(root, "project-b", "s2", [turn("skill-audit")])
        self.assertIn("Scanned 2 transcript file(s)", self.run_script(
            "--transcripts", str(root)
        ).stdout)
        self.assertEqual(self.envelope(root)["transcripts"], 2)

    def test_the_window_label_matches_the_mode(self):
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("skill-audit")])
        self.assertEqual(self.envelope(root, "--all")["window"], "Every transcript")
        self.assertEqual(self.envelope(root, "--since", "14")["window"], "Window: last 14 days")

    def test_rows_are_grouped_by_kind_and_best_evidenced_first(self):
        root = self.tmpdir()
        self.write_transcript(
            root, "project-a", "s1", [turn("writing-plans")] + [turn("skill-audit")] * 3
        )
        assets = json.loads(
            self.run_script("--transcripts", str(root), "--json").stdout
        )["assets"]
        kinds = [row["kind"] for row in assets]
        self.assertEqual(kinds, sorted(kinds))
        skills = [row for row in assets if row["kind"] == "skill"]
        self.assertEqual([r["name"] for r in skills[:2]], ["skill-audit", "writing-plans"])
        self.assertEqual([r["mentions"] for r in skills[:2]], [3, 1])

    def test_every_claude_skill_agent_and_command_gets_a_row(self):
        root = self.tmpdir()
        self.write_transcript(root, "project-a", "s1", [turn("x")])
        rows = {(row["kind"], row["name"]) for row in self.envelope(root)["assets"]}
        for path in (REPO / ".claude" / "skills").glob("*/SKILL.md"):
            self.assertIn(("skill", path.parent.name), rows)
        for path in (REPO / ".claude" / "agents").glob("*.md"):
            self.assertIn(("agent", path.stem), rows)
        for path in (REPO / ".claude" / "commands").glob("*.md"):
            self.assertIn(("command", path.stem), rows)


if __name__ == "__main__":
    unittest.main()
