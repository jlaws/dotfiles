"""Tests for macos_setup.brew."""

import unittest

from macos_setup.brew import install_packages
from macos_setup.shell import CompletedResult
from tests.fakes import FakeRunner


def _brew_ok(argv):
    if argv == ["brew", "--version"]:
        return CompletedResult(0, "Homebrew 4.0.0", "")
    if argv == ["brew", "--prefix"]:
        return CompletedResult(0, "/opt/homebrew\n", "")
    return CompletedResult(0, "", "")


class InstallPackagesTests(unittest.TestCase):
    def test_raises_when_brew_missing(self):
        runner = FakeRunner(lambda argv: CompletedResult(1, "", "not found"))
        with self.assertRaises(RuntimeError):
            install_packages(runner)

    def test_missing_brew_logs_error(self):
        runner = FakeRunner(lambda argv: CompletedResult(1, "", "not found"))
        with self.assertLogs("macos_setup.brew", level="ERROR"):
            with self.assertRaises(RuntimeError):
                install_packages(runner)

    def test_runs_update_install_and_cleanup_in_order(self):
        runner = FakeRunner(_brew_ok)
        install_packages(runner)

        argvs = runner.argv_list()
        self.assertIn(["brew", "update"], argvs)
        self.assertIn(["brew", "install", "coreutils"], argvs)
        self.assertIn(["brew", "cleanup"], argvs)
        self.assertLess(argvs.index(["brew", "update"]), argvs.index(["brew", "install", "coreutils"]))
        self.assertLess(argvs.index(["brew", "install", "coreutils"]), argvs.index(["brew", "cleanup"]))

    def test_installs_fetch_tool_clis(self):
        runner = FakeRunner(_brew_ok)
        install_packages(runner)

        argvs = runner.argv_list()
        self.assertIn(["brew", "install", "poppler"], argvs)
        self.assertIn(["npm", "install", "-g", "agent-browser"], argvs)

    def test_install_logs_info_per_package(self):
        runner = FakeRunner(_brew_ok)
        with self.assertLogs("macos_setup.brew", level="INFO") as cm:
            install_packages(runner)
        self.assertTrue(any("coreutils" in line for line in cm.output))

    def test_dry_run_installs_nothing(self):
        runner = FakeRunner(_brew_ok)
        install_packages(runner, dry_run=True)

        self.assertNotIn(["brew", "install", "coreutils"], runner.argv_list())


if __name__ == "__main__":
    unittest.main()
