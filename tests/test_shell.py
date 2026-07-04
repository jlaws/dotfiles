"""Tests for macos_setup.shell."""

import subprocess
import unittest

from macos_setup.shell import CompletedResult, Runner, build_argv


class BuildArgvTests(unittest.TestCase):
    def test_no_sudo_returns_argv_unchanged(self):
        self.assertEqual(build_argv(["defaults", "read"], sudo=False), ["defaults", "read"])

    def test_sudo_prepends_sudo(self):
        self.assertEqual(build_argv(["pmset", "-a"], sudo=True), ["sudo", "pmset", "-a"])


class RunnerLoggingTests(unittest.TestCase):
    def test_logs_command_and_exit_code_at_debug(self):
        with self.assertLogs("macos_setup.shell", level="DEBUG") as cm:
            Runner().run(["printf", "hi"], capture=True)
        joined = " ".join(cm.output)
        self.assertIn("printf", joined)
        self.assertIn("exit 0", joined)

    def test_logs_sudo_prefix(self):
        with self.assertLogs("macos_setup.shell", level="DEBUG") as cm:
            Runner().run(["true"], sudo=False)
        self.assertNotIn("sudo", " ".join(cm.output))


class RunnerTests(unittest.TestCase):
    def test_capture_returns_stdout(self):
        result = Runner().run(["printf", "hi"], capture=True)
        self.assertIsInstance(result, CompletedResult)
        self.assertEqual(result.stdout, "hi")
        self.assertEqual(result.returncode, 0)

    def test_check_raises_on_nonzero(self):
        with self.assertRaises(subprocess.CalledProcessError):
            Runner().run(["false"], check=True)

    def test_no_check_returns_nonzero_without_raising(self):
        result = Runner().run(["false"], check=False)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
