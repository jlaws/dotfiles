"""Tests for macos_setup.brew."""

from __future__ import annotations

import unittest

from macos_setup.brew import brew_writable, install_packages
from macos_setup.shell import CompletedResult
from tests.fakes import FakeRunner


def _brew_ok(argv):
    if argv == ["brew", "--version"]:
        return CompletedResult(0, "Homebrew 4.0.0", "")
    if argv == ["brew", "--prefix"]:
        return CompletedResult(0, "/opt/homebrew\n", "")
    if argv == ["brew", "--repository"]:
        return CompletedResult(0, "/opt/homebrew\n", "")
    return CompletedResult(0, "", "")


def _brew_readonly(argv):
    """Homebrew installed by an admin; the current (standard) user cannot write to it."""
    if argv == ["test", "-w", "/opt/homebrew"]:
        return CompletedResult(1, "", "")
    return _brew_ok(argv)


class InstallPackagesTests(unittest.TestCase):
    def test_missing_brew_skips_with_warning(self):
        runner = FakeRunner(lambda argv: CompletedResult(1, "", "not found"))
        with self.assertLogs("macos_setup.brew", level="WARNING"):
            self.assertFalse(install_packages(runner))
        self.assertFalse(any(argv[:2] == ["brew", "install"] for argv in runner.argv_list()))

    def test_unwritable_brew_skips_with_warning(self):
        """A standard (non-admin) user can run `brew --version` but not `brew install`."""
        runner = FakeRunner(_brew_readonly)
        with self.assertLogs("macos_setup.brew", level="WARNING") as cm:
            self.assertFalse(install_packages(runner))
        self.assertTrue(any("admin" in line for line in cm.output))
        argvs = runner.argv_list()
        self.assertFalse(any(argv[:2] in (["brew", "install"], ["brew", "update"]) for argv in argvs))
        self.assertNotIn(["npm", "install", "-g", "typescript-language-server", "typescript"], argvs)

    def test_brew_writable_true_for_admin(self):
        self.assertTrue(brew_writable(FakeRunner(_brew_ok)))

    def test_install_packages_never_uses_sudo(self):
        runner = FakeRunner(_brew_ok)
        install_packages(runner)
        self.assertFalse(any(call["sudo"] for call in runner.calls))

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

    def test_installs_jq_because_the_prompt_log_hook_requires_it(self):
        """`log-prompt.sh` parses its stdin payload with `jq` to read `.prompt`. It fails silently
        on a machine without it, so setup has to install it rather than assume it.
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

    def test_a_failed_chrome_download_does_not_abort_the_rest_of_the_install(self):
        """`agent-browser install` is the one bootstrap step that pulls a large binary over the
        network, and it runs ahead of npm and `brew cleanup`. A flaky download must
        not take the npm language servers or `brew cleanup` with it, so it passes check=False.
        """

        def handler(argv):
            if argv == ["agent-browser", "install"]:
                return CompletedResult(1, "", "network unreachable")
            return None

        runner = FakeRunner(lambda argv: handler(argv) or _brew_ok(argv))
        install_packages(runner, dry_run=False)

        argvs = runner.argv_list()
        self.assertIn(["agent-browser", "install"], argvs)
        # Everything sequenced after it still ran.
        self.assertIn(["npm", "install", "-g", "typescript-language-server", "typescript"], argvs)
        self.assertIn(["brew", "cleanup"], argvs)

    def test_installs_no_rust_or_lean_tooling(self):
        """Setup installs no rustup, rust-analyzer, or elan; projects pin their own toolchains.

        A brew rust-analyzer or rustup would also shadow the toolchain's own copy on PATH.
        """
        import macos_setup.brew as brew

        self.assertFalse(hasattr(brew, "install_user_tooling"))
        runner = FakeRunner(_brew_ok)
        install_packages(runner)
        commands = [" ".join(argv) for argv in runner.argv_list()]
        for name in ("rustup", "rust-analyzer", "elan", "sh.rustup.rs", "cargo"):
            self.assertFalse(any(name in command for command in commands), name)

    def test_does_not_install_ai_agent_clis(self) -> None:
        """Claude Code, Codex, and Antigravity CLIs are installed by the user, not by setup."""
        runner = FakeRunner(_brew_ok)
        install_packages(runner)

        commands = [" ".join(argv) for argv in runner.argv_list()]
        for name in ("claude.ai/install", "codex", "antigravity"):
            self.assertFalse(any(name in command for command in commands), name)

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
