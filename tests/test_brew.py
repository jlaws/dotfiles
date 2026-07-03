"""Tests for macos_setup.brew."""

import unittest

from macos_setup.brew import install_packages
from macos_setup.shell import CompletedResult
from tests.fakes import FakeRunner


def _noop(*_args):
    pass


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
            install_packages(runner, log=_noop)

    def test_runs_update_install_and_cleanup_in_order(self):
        runner = FakeRunner(_brew_ok)
        install_packages(runner, log=_noop)

        argvs = runner.argv_list()
        self.assertIn(["brew", "update"], argvs)
        self.assertIn(["brew", "install", "coreutils"], argvs)
        self.assertIn(["brew", "cleanup"], argvs)
        self.assertLess(argvs.index(["brew", "update"]), argvs.index(["brew", "install", "coreutils"]))
        self.assertLess(argvs.index(["brew", "install", "coreutils"]), argvs.index(["brew", "cleanup"]))

    def test_dry_run_installs_nothing(self):
        runner = FakeRunner(_brew_ok)
        install_packages(runner, dry_run=True, log=_noop)

        self.assertNotIn(["brew", "install", "coreutils"], runner.argv_list())


if __name__ == "__main__":
    unittest.main()
