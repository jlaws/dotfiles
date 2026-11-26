"""Exercise the bundled test-pollution helper without running npm."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HELPERS = tuple(
    REPO / tree / "skills/debugging-methodology/find-polluter.sh"
    for tree in (".agents", ".claude", ".gemini/antigravity-cli")
)


class FindPolluterTests(unittest.TestCase):
    def run_helper(
        self, helper: Path, root: Path, npm_status: int = 0
    ) -> subprocess.CompletedProcess[str]:
        executable = root / "bin/npm"
        executable.parent.mkdir(exist_ok=True)
        executable.write_text(
            '#!/bin/bash\nprintf "%s\\n" "$@" >> calls.txt\n' + f"exit {npm_status}\n"
        )
        executable.chmod(0o755)
        env = dict(os.environ, PATH=f"{executable.parent}:{os.environ['PATH']}")
        return subprocess.run(
            ["bash", str(helper), "pollution", "src/**/*.test.ts"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=10,
        )

    def test_no_matches_is_not_a_clean_run(self):
        for helper in HELPERS:
            with self.subTest(helper=helper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.run_helper(helper, root)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("Found 0 test files", result.stdout)
                self.assertIn("No test files matched", result.stdout)
                self.assertNotIn("all tests clean", result.stdout)
                self.assertFalse((root / "calls.txt").exists())

    def test_preexisting_target_is_inconclusive_without_running_tests(self):
        for helper in HELPERS:
            with self.subTest(helper=helper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "src").mkdir()
                (root / "src/direct.test.ts").touch()
                (root / "pollution").mkdir()
                result = self.run_helper(helper, root)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("Pollution already exists", result.stdout)
                self.assertNotIn("all tests clean", result.stdout)
                self.assertFalse((root / "calls.txt").exists())

    def test_failed_test_command_is_inconclusive_without_pollution(self):
        for helper in HELPERS:
            with self.subTest(helper=helper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "src").mkdir()
                (root / "src/direct.test.ts").touch()
                result = self.run_helper(helper, root, npm_status=7)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("Test command failed (exit 7)", result.stdout)
                self.assertNotIn("all tests clean", result.stdout)
                self.assertTrue((root / "calls.txt").is_file())

    def test_relative_glob_runs_direct_and_nested_matches_with_spaces(self):
        for helper in HELPERS:
            with self.subTest(helper=helper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = ["src/direct.test.ts", "src/nested dir/deeper/matched.test.ts"]
                for path in paths:
                    file = root / path
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.touch()
                result = self.run_helper(helper, root)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("Found 2 test files", result.stdout)
                calls = (root / "calls.txt").read_text().splitlines()
                self.assertEqual([line for line in calls if line.endswith(".test.ts")], paths)
