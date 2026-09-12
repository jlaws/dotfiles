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
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLAUDE_HOOK = REPO / ".claude" / "hooks" / "guard-bash-output.sh"
CODEX_HOOK = REPO / ".codex" / "hooks" / "guard-bash-output.sh"


def setUpModule():
    """Require jq rather than skipping without it.

    These tests used to carry `@unittest.skipUnless(shutil.which("jq"), ...)`. On a machine without
    jq that reported `OK (skipped=27)` while the guard was 100% dead -- a green `make verify` on a
    broken feature. macOS ships jq at /usr/bin/jq, so requiring it costs nothing on the supported
    platform, and the guard's own jq-missing path is covered explicitly below instead.
    """
    if not shutil.which("jq"):
        raise RuntimeError("jq is required: the Bash guard shells out to it")


def run_hook(
    command: str, fmt: str = "claude", hook: Path = CLAUDE_HOOK, env: "dict[str, str] | None" = None
) -> "subprocess.CompletedProcess[str]":
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    return subprocess.run(
        ["bash", str(hook), "--format", fmt],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )


def message(command: str, fmt: str = "claude") -> str:
    """Return the advisory text the hook emitted, or "" when it stayed silent.

    Asserts the hook exited 0 with a clean stderr first. Reading stdout alone made a crash
    indistinguishable from silence: a hook that died mid-rule produced empty stdout, and every
    silence assertion in this file read that as a pass.
    """
    proc = run_hook(command, fmt)
    if proc.returncode != 0 or proc.stderr:
        raise AssertionError(
            f"hook failed for {command!r}: rc={proc.returncode} stderr={proc.stderr!r}"
        )
    out = proc.stdout.strip()
    if not out:
        return ""
    payload = json.loads(out)
    if fmt == "claude":
        return payload["systemMessage"]
    return payload["hookSpecificOutput"]["additionalContext"]


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
                self.assertEqual(run_hook(command).returncode, 0)

    def test_never_blocks_and_never_rewrites(self):
        for fmt in ("claude", "codex"):
            proc = run_hook("git log", fmt)
            with self.subTest(fmt=fmt):
                self.assertEqual(proc.returncode, 0)
                self.assertEqual(proc.stderr, "")
                self.assertNotIn("updatedInput", proc.stdout)
                self.assertNotIn("permissionDecision", proc.stdout)

    def test_claude_format_uses_top_level_system_message(self):
        payload = json.loads(run_hook("git log", "claude").stdout)
        self.assertIn("--oneline", payload["systemMessage"])
        self.assertNotIn("hookSpecificOutput", payload)

    def test_codex_format_nests_additional_context(self):
        payload = json.loads(run_hook("git log", "codex").stdout)
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
        proc = run_hook('git log --format="%h `date`"')
        self.assertEqual(proc.returncode, 0)
        # Unconditionally, not `if out:`. Guarding the parse on non-empty output made the test
        # vacuous the moment a regression silenced the rule -- a silencing bug and a JSON-encoding
        # bug became indistinguishable.
        self.assertIn("--oneline", json.loads(proc.stdout)["systemMessage"])


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


class BashGuardPrecisionTests(unittest.TestCase):
    """Boundary cases the first cut of the guard got wrong, in both directions.

    A guard that nags about bounded commands trains the reader to ignore it, and one that stays
    silent on the highest-volume unbounded command does not earn its place in every Bash call. Each
    case here was reproduced against the live hook before being written down.
    """

    def assert_flags(self, command: str, expected: str) -> None:
        msg = message(command)
        self.assertTrue(msg, f"expected a suggestion for {command!r}, got silence")
        self.assertIn(expected, msg)

    def assert_silent(self, command: str) -> None:
        self.assertEqual(message(command), "", f"expected silence for {command!r}")

    def test_bounded_flags_in_attached_form_are_silent(self):
        """`-n20` and `-L2` are the ordinary way to write these; only the spaced form was matched."""
        for command in ["git log -n20", "git log --max-count=20", "tree -L2", "grep -m5 TODO ."]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_revision_arguments_do_not_count_as_path_scope(self):
        """`git diff main...HEAD` is what this repo's own j-diff-review tells agents to run, and it
        is the largest diff the guard will ever see. A revision is not a path scope.
        """
        for command in ["git diff main...HEAD", "git diff HEAD", "git diff main", "git diff @~2"]:
            with self.subTest(command=command):
                self.assert_flags(command, "--stat")

    def test_path_scope_still_silences_git_diff(self):
        for command in ["git diff macos_setup/brew.py", "git diff -- tests/", "git diff --stat"]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_only_truly_bounding_pipes_silence_a_rule(self):
        """sed/awk/sort/uniq/cut emit one line per input line. They transform; they do not bound."""
        for command in ["cat big.log | sort", "cat f | cut -c1-80", "cat f.txt | sed s/a/b/"]:
            with self.subTest(command=command):
                self.assert_flags(command, "Read")

    def test_head_tail_and_wc_do_silence_a_rule(self):
        for command in ["cat big.log | head -20", "cat f | tail -5", "cat f | wc -l"]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_every_segment_of_a_chain_is_inspected(self):
        """Regression, both directions. The first cut nagged about `&&` itself, which fired on
        nearly every command and, being first in the chain, shadowed all seven output rules.
        Deleting that rule then left every rule anchored at `^`, so `cd . &&` became a universal
        opt-out. Neither is right: split the command and run the ladder on each segment.
        """
        for command in [
            "cd /tmp && ls -R",
            "make check && cat huge.log",
            "git status --porcelain; cat huge.log",
            "make build || cat error.log",
        ]:
            with self.subTest(command=command):
                self.assertTrue(message(command), f"expected a suggestion for {command!r}")

    def test_a_clean_chain_stays_silent(self):
        """Chaining is not itself the defect; unbounded output is."""
        for command in ["cd /tmp && ls", "git add -A && git commit -m x", "make check && make test"]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_separators_inside_quotes_are_not_separators(self):
        """A search pattern is an operand, not a pipeline."""
        for command in ["grep -E 'head|tail' file.txt", "grep 'foo|head' ."]:
            with self.subTest(command=command):
                self.assert_flags(command, "Grep")

    def test_flag_lookalikes_inside_quotes_do_not_silence_a_rule(self):
        for command in ['grep -rn "needle -l here" .', 'git log --grep="revert -20 thing"']:
            with self.subTest(command=command):
                self.assertTrue(message(command), f"expected a suggestion for {command!r}")

    def test_test_runners_need_a_scope_not_merely_an_argument(self):
        """`-v` is not a scope. The predicate is "has a scoping argument", not "has an argument"."""
        for command in ["pytest -v", "pytest -q --tb=short", "go test -v ./...", "cargo test -- --nocapture"]:
            with self.subTest(command=command):
                self.assert_flags(command, "scope")

    def test_a_real_scope_silences_a_test_runner(self):
        for command in ["pytest tests/test_brew.py", "pytest -v -k parser", "go test ./macos_setup"]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_long_option_recursive_listing_is_flagged(self):
        self.assert_flags("ls --recursive", "Recursive")

    def test_a_trailing_comment_cannot_disable_the_guard(self):
        """`<<` anywhere used to kill the guard outright, so `# <<` was a one-token opt-out."""
        self.assert_flags("cat bigfile.txt # <<", "Read")

    def test_real_heredocs_are_still_exempt(self):
        for command in ["cat > /tmp/x.md <<'EOF'\nbody\nEOF", "cat <<EOF\nx\nEOF"]:
            with self.subTest(command=command):
                self.assert_silent(command)

    def test_codex_advice_names_no_claude_only_tool(self):
        """The same bytes ship to Codex, which has no Read/Grep/Glob tool. Naming one there is
        advice the reader cannot act on -- the same reason rung 4 of the search ladder was reworded
        for the shared tree.
        """
        for command in ["cat big.log", "find .", "grep -rn TODO ."]:
            with self.subTest(command=command):
                msg = message(command, fmt="codex")
                self.assertTrue(msg, f"expected a suggestion for {command!r}")
                for tool in ("Read tool", "Grep tool", "Glob tool"):
                    self.assertNotIn(tool, msg)

    def test_advisory_text_never_echoes_the_command_back(self):
        """The first cut classified with a regex prefix then interpolated the matched token into
        the advisory. `[[:space:]]` matches 11 characters that IFS does not split on, so a command
        like `grep<NBSP><ANSI escape>` had its payload re-emitted through `systemMessage` -- a
        channel the model reads as harness-authored rather than as tool input. Matching the token
        exactly closes it; this pins that it stays closed.
        """
        for payload in ["grep \u00a0\x1b]0;PWNED\x07", "grep\r\x1b[2J x", "rg \u2009<!--OVERRIDE-->"]:
            with self.subTest(payload=payload):
                msg = message(payload)
                for bad in ("\x1b", "\r", "\x07", "PWNED", "OVERRIDE"):
                    self.assertNotIn(bad, msg)

    def test_an_unknown_format_stays_silent(self):
        """A `--format codx` typo must not hand Codex a field it does not read."""
        for fmt in ("bogus", "codx", ""):
            with self.subTest(fmt=fmt):
                self.assertEqual(run_hook("git log", fmt).stdout.strip(), "")


class _TmpDirMixin(unittest.TestCase):
    def _tmpdir(self) -> str:
        """TemporaryDirectory bound to the test's lifetime.

        `enterContext` would be the obvious call, but it is 3.11+ and this repo type-checks against
        python-version = "3.9".
        """
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return td.name


class BashGuardDegradedModeTests(_TmpDirMixin):
    """The fail-open path, exercised rather than assumed.

    This was the guard's single most important property and had zero coverage, because the only
    test that could have reached it was gated on the runner's own jq.
    """

    def _run_without_jq(self, command: str) -> "subprocess.CompletedProcess[str]":
        bindir = Path(self._tmpdir())
        for tool in ("bash", "cat", "awk", "grep", "printf"):
            found = shutil.which(tool)
            if found:
                (bindir / tool).symlink_to(found)
        env = {"PATH": str(bindir), "HOME": os.environ.get("HOME", "/tmp")}
        return run_hook(command, env=env)

    def test_missing_jq_is_inert_but_not_invisible(self):
        proc = self._run_without_jq("git log")
        self.assertEqual(proc.returncode, 0, "a missing jq must never fail the tool call")
        self.assertEqual(proc.stdout, "", "no jq means no advisory")
        self.assertIn(
            "jq not found",
            proc.stderr,
            "a silent bail makes an inert guard byte-identical to a healthy one",
        )


class PromptLogRedactionTests(_TmpDirMixin):
    """The prompt log had no tests at all, and PR 95 made it live on a second harness.

    Every vector below was reproduced against the previous rules, which anchored on uppercase
    SECRET/TOKEN/KEY with `=`. That missed lowercase `api_key=`, every `password=`, every colon
    form, and every provider-prefixed token.
    """

    def _log(self, prompt: str) -> "tuple[str, Path]":
        tmp = Path(self._tmpdir())
        logfile = tmp / "PROMPT_LOG.md"
        env = dict(os.environ, PROMPT_LOG_FILE=str(logfile))
        proc = subprocess.run(
            ["bash", str(REPO / ".claude" / "hooks" / "log-prompt.sh")],
            input=json.dumps({"prompt": prompt}),
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return logfile.read_text(), logfile

    def test_secret_shapes_are_redacted(self):
        vectors = [
            ("AKIAIOSFODNN7EXAMPLE", "IOSFODNN7EXAMPLE"),
            ("ghp_16C7e42F292c6912E7710c838347Ae178B4a", "e42F292c6912E7710c838347Ae178B4a"),
            ("api_key=hunter2", "hunter2"),
            ("password=hunter2", "hunter2"),
            ("DB_PASSWORD=hunter2", "hunter2"),
            ("API_KEY: huntercolon", "huntercolon"),
            ("postgres://admin:S3cr3tP4ss@db/prod", "S3cr3tP4ss"),
            ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
            ("xoxb-123456789012-1234567890123-AbCdEf", "1234567890123"),
            ("eyJhbGciOiJIUzI1NiJ9.abc.def", "OiJIUzI1NiJ9"),
        ]
        for prompt, leaked in vectors:
            with self.subTest(prompt=prompt):
                body, _ = self._log(prompt)
                self.assertNotIn(leaked, body)
                self.assertIn("**REDACTED**", body)

    def test_ordinary_prose_is_left_alone(self):
        """Over-redaction would make the log useless, so the rules must not fire on plain text."""
        body, _ = self._log("please explain the key differences between these two approaches")
        self.assertIn("key differences", body)
        self.assertNotIn("**REDACTED**", body)

    def test_the_log_is_not_world_readable(self):
        """Codex keeps auth.json and history.jsonl at 0600 in the same directory."""
        _, logfile = self._log("hello")
        self.assertEqual(oct(logfile.stat().st_mode & 0o777), "0o600")

    def test_a_missing_parent_directory_does_not_stall(self):
        """mkdir of the lock cannot succeed under a missing parent, so the retry loop used to burn
        its full budget -- 5.0s, exactly the hook timeout -- on what is really a path error.
        """
        tmp = Path(self._tmpdir())
        logfile = tmp / "deep" / "nested" / "PROMPT_LOG.md"
        env = dict(os.environ, PROMPT_LOG_FILE=str(logfile))
        start = time.monotonic()
        proc = subprocess.run(
            ["bash", str(REPO / ".claude" / "hooks" / "log-prompt.sh")],
            input=json.dumps({"prompt": "hi"}),
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        elapsed = time.monotonic() - start
        self.assertEqual(proc.returncode, 0)
        self.assertLess(elapsed, 2.0, f"took {elapsed:.2f}s; the lock loop is stalling")
        self.assertIn("hi", logfile.read_text())

    def test_the_log_rotates_instead_of_growing_without_bound(self):
        """Unrotated, this file reached 110 MB in about six months on the author's machine."""
        tmp = Path(self._tmpdir())
        logfile = tmp / "PROMPT_LOG.md"
        logfile.write_text("x" * 4096)
        env = dict(os.environ, PROMPT_LOG_FILE=str(logfile), PROMPT_LOG_MAX_BYTES="1024")
        subprocess.run(
            ["bash", str(REPO / ".claude" / "hooks" / "log-prompt.sh")],
            input=json.dumps({"prompt": "after rotation"}),
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        self.assertTrue(logfile.with_suffix(".md.1").exists(), "previous generation not kept")
        self.assertIn("after rotation", logfile.read_text())
        self.assertNotIn("xxxx", logfile.read_text())

    def test_both_trees_ship_the_same_logger_apart_from_its_path(self):
        claude = (REPO / ".claude" / "hooks" / "log-prompt.sh").read_text()
        codex = (REPO / ".codex" / "hooks" / "log-prompt.sh").read_text()
        self.assertEqual(claude.replace("/.claude/", "/.codex/"), codex)


if __name__ == "__main__":
    unittest.main()
