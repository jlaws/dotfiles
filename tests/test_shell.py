"""Tests for macos_setup.shell."""

import subprocess
import unittest

from macos_setup.shell import CompletedResult, Runner, build_argv


class BuildArgvTests(unittest.TestCase):
    def test_no_sudo_returns_argv_unchanged(self):
        self.assertEqual(build_argv(["defaults", "read"], sudo=False), ["defaults", "read"])

    def test_sudo_prepends_sudo(self):
        self.assertEqual(build_argv(["pmset", "-a"], sudo=True), ["sudo", "pmset", "-a"])


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
