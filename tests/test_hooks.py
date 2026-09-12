"""Behaviour tests for the PreToolUse Bash guard.

The guard is ADVISORY. It never blocks a command and never rewrites one. A hook that edits the
command an agent asked for puts a lossy layer between the agent and its evidence, contradicting
`CLAUDE.md`'s byte-for-byte rule. This is the Fail-Open Principle already stated in
`references/workflow/hook-patterns.md`, applied: it only ever emits an advisory message and exits 0.

Claude and Codex read that message from different fields, so the script takes `--format`:
Claude uses a top-level `systemMessage`, Codex uses `hookSpecificOutput.additionalContext`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLAUDE_HOOK = REPO / ".claude" / "hooks" / "guard-bash-output.sh"
CODEX_HOOK = REPO / ".codex" / "hooks" / "guard-bash-output.sh"


def run_hook(command: str, fmt: str = "claude", hook: Path = CLAUDE_HOOK) -> "tuple[int, str]":
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    proc = subprocess.run(
        ["bash", str(hook), "--format", fmt],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.returncode, proc.stdout


def message(command: str, fmt: str = "claude") -> str:
    """Return the advisory text the hook emitted, or "" when it stayed silent."""
    out = run_hook(command, fmt)[1].strip()
    if not out:
        return ""
    payload = json.loads(out)
    if fmt == "claude":
        return payload["systemMessage"]
    return payload["hookSpecificOutput"]["additionalContext"]


@unittest.skipUnless(shutil.which("jq"), "jq is required by the hook")
class BashGuardContractTests(unittest.TestCase):
    def test_both_trees_ship_the_same_executable_hook(self):
        for hook in (CLAUDE_HOOK, CODEX_HOOK):
            with self.subTest(hook=hook.relative_to(REPO)):
                self.assertTrue(hook.is_file())
                self.assertTrue(os.access(hook, os.X_OK), "hook must be mode 755")
        self.assertEqual(
            CLAUDE_HOOK.read_bytes(),
            CODEX_HOOK.read_bytes(),
            "the two copies must stay byte-identical; only --format differs at call time",
        )

    def test_shebang_matches_the_registered_interpreter(self):
        # settings.json and config.toml both register `bash <path>`. A zsh shebang, as
        # log-prompt.sh carries, would mean these tests prove nothing about the live path.
        self.assertEqual(CLAUDE_HOOK.read_text().splitlines()[0], "#!/usr/bin/env bash")

    def test_always_exits_zero(self):
        for command in ["git log", "ls", "", "git log --oneline -20", "cat <<EOF\nx\nEOF"]:
            with self.subTest(command=command):
                self.assertEqual(run_hook(command)[0], 0)

    def test_never_blocks_and_never_rewrites(self):
        for fmt in ("claude", "codex"):
            code, out = run_hook("git log", fmt)
            with self.subTest(fmt=fmt):
                self.assertEqual(code, 0)
                self.assertNotIn("updatedInput", out)
                self.assertNotIn("permissionDecision", out)

    def test_claude_format_uses_top_level_system_message(self):
        payload = json.loads(run_hook("git log", "claude")[1])
        self.assertIn("--oneline", payload["systemMessage"])
        self.assertNotIn("hookSpecificOutput", payload)

    def test_codex_format_nests_additional_context(self):
        payload = json.loads(run_hook("git log", "codex")[1])
        self.assertEqual(payload["hookSpecificOutput"]["hookEventName"], "PreToolUse")
        self.assertIn("--oneline", payload["hookSpecificOutput"]["additionalContext"])
        self.assertNotIn("systemMessage", payload)

    def test_defaults_to_claude_format_without_a_flag(self):
        payload = json.loads(
            subprocess.run(
                ["bash", str(CLAUDE_HOOK)],
                input=json.dumps({"tool_input": {"command": "git log"}}),
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        )
        self.assertIn("systemMessage", payload)

    def test_survives_malformed_and_empty_input(self):
        for payload in ["not json", "", "{}", '{"tool_input": {}}']:
            with self.subTest(payload=payload):
                proc = subprocess.run(
                    ["bash", str(CLAUDE_HOOK)],
                    input=payload,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(proc.returncode, 0)
                self.assertEqual(proc.stdout.strip(), "")

    def test_quotes_and_backticks_survive_json_encoding(self):
        # The suggestion is built with `jq -n --arg`, so a command carrying quotes must not
        # produce invalid JSON.
        code, out = run_hook('git log --format="%h `date`"')
        self.assertEqual(code, 0)
        if out.strip():
            json.loads(out)


@unittest.skipUnless(shutil.which("jq"), "jq is required by the hook")
class BashGuardRuleTests(unittest.TestCase):
    def assert_flags(self, command: str, expected: str) -> None:
        msg = message(command)
        self.assertTrue(msg, f"expected a suggestion for {command!r}, got silence")
        self.assertIn(expected, msg)

    def assert_silent(self, command: str) -> None:
        self.assertEqual(message(command), "", f"expected silence for {command!r}")

    def test_unbounded_git_log(self):
        self.assert_flags("git log", "--oneline")

    def test_bounded_git_log_is_silent(self):
        for command in ["git log --oneline -20", "git log -n 5", "git log -3", "git log --stat"]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_unbounded_git_diff(self):
        self.assert_flags("git diff", "--stat")

    def test_bounded_git_diff_is_silent(self):
        for command in ["git diff --stat", "git diff --name-only", "git diff macos_setup/brew.py"]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_bare_cat(self):
        self.assert_flags("cat macos_setup/brew.py", "Read")

    def test_piped_cat_is_silent(self):
        for command in ["cat f.json | jq .", "cat f.txt | grep x", "cat f.txt | head -20"]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_unbounded_find(self):
        self.assert_flags("find . -type f", "Glob")

    def test_bounded_find_is_silent(self):
        for command in ["find . -maxdepth 2 -type f", "find . -name '*.py'"]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_unbounded_grep(self):
        self.assert_flags("grep -r pattern .", "Grep")

    def test_bounded_grep_is_silent(self):
        for command in ["grep -l pattern .", "grep -c pattern f", "grep -m 5 pattern f"]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_recursive_listing(self):
        self.assert_flags("ls -R", "-L 2")
        self.assert_flags("tree", "-L 2")

    def test_bounded_listing_is_silent(self):
        for command in ["ls -1", "tree -L 2"]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_unscoped_test_run(self):
        for command in ["pytest", "cargo test", "npm test", "go test ./..."]:
            with self.subTest(command=command):
                self.assert_flags(command, "scope")

    def test_scoped_test_run_is_silent(self):
        for command in ["pytest tests/test_brew.py", "cargo test parser::", "npm test -- --grep x"]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_chained_commands_are_silent(self):
        """Regression: an `&&` rule here fired on nearly every command an agent writes, and being
        first in the chain it shadowed all seven output rules. The guard bounds output size; failure
        attribution is a separate concern, already stated in prose in CLAUDE.md's Bash section and
        loaded every turn. Restating it per command was noise, not signal.
        """
        for command in [
            "git add -A && git commit -m x",
            "make check && make test",
            "cd /tmp && ls",
        ]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_single_command_is_silent(self):
        self.assert_silent("git status --porcelain")

    def test_heredoc_is_never_flagged(self):
        # Report bodies and file writes use quoted heredocs; flagging them would be noise.
        self.assert_silent("cat > /tmp/x.md <<'EOF'\nbody\nEOF")

    def test_already_compact_commands_are_silent(self):
        for command in ["git status --porcelain", "git rev-parse HEAD", "wc -l f", "echo hi"]:
            with self.subTest(command=command):
                self.assert_silent(command)


if __name__ == "__main__":
    unittest.main()
