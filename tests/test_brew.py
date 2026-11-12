"""Tests for macos_setup.brew."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from macos_setup.brew import brew_writable, install_packages, install_user_tooling
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
    def setUp(self) -> None:
        self._orig_path = os.environ.get("PATH")

    def tearDown(self) -> None:
        if self._orig_path is not None:
            os.environ["PATH"] = self._orig_path
        else:
            os.environ.pop("PATH", None)

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

    def test_user_tooling_needs_no_homebrew(self):
        """rustup, elan, and the agent CLIs install under ~, so a standard user gets them."""
        runner = FakeRunner()
        install_user_tooling(runner)
        argvs = runner.argv_list()
        self.assertFalse(any(argv[0] in ("brew", "npm") for argv in argvs))
        self.assertFalse(any(call["sudo"] for call in runner.calls))
        self.assertIn(["rustup", "default", "stable"], argvs)

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

    def test_regression_rust_analyzer_comes_from_rustup_not_brew(self):
        """rust-analyzer must come from `rustup component add`, never `brew install rust-analyzer`.

        A brew-installed rust-analyzer shadows the toolchain's own copy on PATH and then drifts
        from the active toolchain. That is the original regression and it still applies.

        The mechanism changed in f15fcf5: rustup used to be a brew formula, invoked through an
        absolute `$(brew --prefix rustup)/bin/rustup` path, and is now the official installer from
        sh.rustup.rs followed by a bare `rustup`. The bare call resolves through PATH
        (`~/.cargo/bin`), which `install_user_tooling` ensures is prepended to `os.environ["PATH"]`.
        This guards against the wrong source for rust-analyzer and guarantees binary resolvability.
        """
        runner = FakeRunner(_brew_ok)
        install_packages(runner)
        install_user_tooling(runner)

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

    def test_ensure_cargo_path_prepends_to_path(self) -> None:
        from macos_setup.brew import _ensure_cargo_path

        fake_home = Path("/fake/home")
        with mock.patch("pathlib.Path.home", return_value=fake_home):
            with mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=True):
                _ensure_cargo_path()
                self.assertEqual(
                    os.environ["PATH"],
                    f"/fake/home/.cargo/bin{os.pathsep}/usr/bin:/bin",
                )

    def test_ensure_cargo_path_respects_cargo_home(self) -> None:
        from macos_setup.brew import _ensure_cargo_path

        with mock.patch.dict(
            os.environ,
            {"CARGO_HOME": "/custom/cargo", "PATH": "/usr/bin"},
            clear=True,
        ):
            _ensure_cargo_path()
            self.assertEqual(
                os.environ["PATH"],
                f"/custom/cargo/bin{os.pathsep}/usr/bin",
            )

    def test_ensure_cargo_path_handles_empty_path(self) -> None:
        from macos_setup.brew import _ensure_cargo_path

        fake_home = Path("/fake/home")
        with mock.patch("pathlib.Path.home", return_value=fake_home):
            with mock.patch.dict(os.environ, {}, clear=True):
                _ensure_cargo_path()
                self.assertEqual(os.environ["PATH"], "/fake/home/.cargo/bin")

    def test_ensure_cargo_path_idempotent(self) -> None:
        from macos_setup.brew import _ensure_cargo_path

        fake_home = Path("/fake/home")
        with mock.patch("pathlib.Path.home", return_value=fake_home):
            with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
                _ensure_cargo_path()
                _ensure_cargo_path()
                self.assertEqual(
                    os.environ["PATH"],
                    f"/fake/home/.cargo/bin{os.pathsep}/usr/bin",
                )

    def test_user_tooling_ensures_cargo_path_before_rustup(self) -> None:
        call_order: list[str] = []

        def record_ensure() -> None:
            call_order.append("ensure_path")

        def handler(argv: list[str]) -> CompletedResult | None:
            if argv and argv[0] == "rustup":
                call_order.append("rustup")
            return _brew_ok(argv)

        runner = FakeRunner(handler)
        with mock.patch("macos_setup.brew._ensure_cargo_path", side_effect=record_ensure):
            install_user_tooling(runner)

        self.assertIn("ensure_path", call_order)
        self.assertIn("rustup", call_order)
        self.assertLess(call_order.index("ensure_path"), call_order.index("rustup"))

    def test_does_not_install_ai_agent_clis(self) -> None:
        """Claude Code, Codex, and Antigravity CLIs are installed by the user, not by setup."""
        runner = FakeRunner(_brew_ok)
        install_user_tooling(runner)
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
