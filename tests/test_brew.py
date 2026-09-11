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
        """The always-loaded configs name a fetch-tool ladder -- WebFetch, then the agent-browser
        CLI for JS-rendered or auth-walled pages, then `pdftotext` for PDFs. Setup has to install
        the two that are not built in, or the guidance points at missing binaries.

        `agent-browser` needs a second step: the brew formula ships the CLI, and
        `agent-browser install` downloads the Chrome binary it drives ("Download Chrome (first
        time)" in its own help). Commit f15fcf5 dropped both, leaving the guidance dangling on a
        fresh Mac; this pins them back.
        """
        runner = FakeRunner(_brew_ok)
        install_packages(runner)

        argvs = runner.argv_list()
        self.assertIn(["brew", "install", "poppler"], argvs)
        self.assertIn(["brew", "install", "agent-browser"], argvs)
        self.assertIn(["agent-browser", "install"], argvs)
        self.assertLess(
            argvs.index(["brew", "install", "agent-browser"]),
            argvs.index(["agent-browser", "install"]),
        )

    def test_installs_jq_because_the_hooks_require_it(self):
        """Every hook parses its stdin payload with `jq`: `log-prompt.sh` reads `.prompt`, and the
        PreToolUse guard reads `.tool_input.command`. Both fail silently on a machine without it,
        so setup has to install it rather than assume it.
        """
        runner = FakeRunner(_brew_ok)
        install_packages(runner)

        self.assertIn(["brew", "install", "jq"], runner.argv_list())

    def test_installs_search_tools(self):
        runner = FakeRunner(_brew_ok)
        install_packages(runner)

        argvs = runner.argv_list()
        self.assertIn(["brew", "install", "fd"], argvs)

    def test_installs_uv(self):
        runner = FakeRunner(_brew_ok)
        install_packages(runner)

        self.assertIn(["brew", "install", "uv"], runner.argv_list())

    def test_does_not_install_go_tooling(self):
        runner = FakeRunner(_brew_ok)
        install_packages(runner)

        argvs = runner.argv_list()
        self.assertNotIn(["brew", "install", "go"], argvs)
        self.assertFalse(any(argv and argv[0] == "go" for argv in argvs))

    def test_regression_rust_analyzer_comes_from_rustup_not_brew(self):
        """rust-analyzer must come from `rustup component add`, never `brew install rust-analyzer`.

        A brew-installed rust-analyzer shadows the toolchain's own copy on PATH and then drifts
        from the active toolchain. That is the original regression and it still applies.

        The mechanism changed in f15fcf5: rustup used to be a brew formula, invoked through an
        absolute `$(brew --prefix rustup)/bin/rustup` path, and is now the official installer from
        sh.rustup.rs followed by a bare `rustup`. The bare call resolves through PATH
        (`~/.cargo/bin`), so this no longer guards against shadowing -- only against the wrong
        source for rust-analyzer, plus the ordering the install depends on.
        """
        runner = FakeRunner(_brew_ok)
        install_packages(runner)

        argvs = runner.argv_list()
        rust_install = ["bash", "-c", "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"]
        self.assertNotIn(["brew", "install", "rust-analyzer"], argvs)
        self.assertNotIn(["brew", "install", "rustup"], argvs)
        self.assertIn(rust_install, argvs)
        self.assertIn(["rustup", "default", "stable"], argvs)
        self.assertIn(["rustup", "component", "add", "rust-analyzer"], argvs)
        self.assertLess(
            argvs.index(rust_install),
            argvs.index(["rustup", "default", "stable"]),
        )
        self.assertLess(
            argvs.index(["rustup", "default", "stable"]),
            argvs.index(["rustup", "component", "add", "rust-analyzer"]),
        )

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
