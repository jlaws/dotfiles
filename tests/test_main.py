"""Tests for macos_setup.__main__ argument resolution and orchestration."""

import contextlib
import hashlib
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from macos_setup.__main__ import build_parser, main, resolve_actions, restart_apps
from macos_setup.archive import Archive
from macos_setup.shell import CompletedResult
from tests.fakes import FakeRunner


def _parse(args):
    return resolve_actions(build_parser().parse_args(args))


class ResolveActionsTests(unittest.TestCase):
    def test_no_flags_runs_all(self):
        actions = _parse([])
        self.assertTrue(actions.dotfiles)
        self.assertTrue(actions.agents)
        self.assertTrue(actions.brew)
        self.assertTrue(actions.macos)
        self.assertTrue(actions.restart)
        self.assertFalse(actions.uninstall)

    def test_single_flag_selects_only_that_step(self):
        actions = _parse(["-d"])
        self.assertTrue(actions.dotfiles)
        self.assertFalse(actions.agents)
        self.assertFalse(actions.brew)
        self.assertFalse(actions.macos)

    def test_grouped_flags(self):
        actions = _parse(["-cb"])
        self.assertTrue(actions.agents)
        self.assertTrue(actions.brew)
        self.assertFalse(actions.dotfiles)

    def test_list_archives_mode(self):
        actions = _parse(["--list-archives"])
        self.assertTrue(actions.list_archives)
        self.assertFalse(actions.dotfiles)

    def test_uninstall_latest(self):
        actions = _parse(["--uninstall"])
        self.assertTrue(actions.uninstall)
        self.assertIsNone(actions.uninstall_ref)

    def test_uninstall_specific_ref(self):
        actions = _parse(["--uninstall", "2026-01-02-030405"])
        self.assertTrue(actions.uninstall)
        self.assertEqual(actions.uninstall_ref, "2026-01-02-030405")

    def test_project_requires_config(self):
        with self.assertRaises(ValueError):
            _parse(["-p", "/tmp/x"])

    def test_uninstall_rejects_install_steps(self):
        with self.assertRaises(ValueError):
            _parse(["--uninstall", "-m"])

    def test_verbose_defaults_false(self):
        self.assertFalse(_parse([]).verbose)

    def test_verbose_flag_parsed(self):
        self.assertTrue(_parse(["-v"]).verbose)
        self.assertTrue(_parse(["--verbose"]).verbose)


class RestartAppsTests(unittest.TestCase):
    def test_logs_info_per_app(self):
        runner = FakeRunner()
        with self.assertLogs("macos_setup.__main__", level="INFO") as cm:
            restart_apps(runner)
        self.assertTrue(any("Dock" in line for line in cm.output))

    def test_dry_run_logs_would_restart(self):
        runner = FakeRunner()
        with self.assertLogs("macos_setup.__main__", level="INFO") as cm:
            restart_apps(runner, dry_run=True)
        self.assertTrue(any("would restart" in line for line in cm.output))


class MainSmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.repo = self.tmp / "repo"
        self.repo.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _main(self, argv, runner):
        """Run main with stdout captured to keep test output clean."""
        with contextlib.redirect_stdout(io.StringIO()):
            return main(argv, runner=runner, home=self.home, repo=self.repo)

    def test_list_archives_empty_returns_zero(self):
        runner = FakeRunner()
        code = self._main(["--list-archives"], runner)
        self.assertEqual(code, 0)

    def test_dry_run_macos_writes_nothing(self):
        runner = FakeRunner(lambda argv: CompletedResult(0, "14.0", ""))
        code = self._main(["-m", "--dry-run", "-f"], runner)

        self.assertEqual(code, 0)
        self.assertFalse(any(c[:2] == ["defaults", "write"] for c in runner.argv_list()))
        self.assertFalse((self.home / ".dotfile-archive").exists())

    def test_uninstall_files_only_skips_sudo_and_restart(self):
        root = self.home / ".dotfile-archive"
        archive = Archive.create(root, "ts1")
        target = self.home / ".zshrc"
        target.write_text("managed")
        archive.record_file(str(target), "added", hashlib.sha256(b"managed").hexdigest())
        archive.save()
        runner = FakeRunner()

        code = self._main(["--uninstall", "-f"], runner)

        argvs = runner.argv_list()
        self.assertEqual(code, 0)
        self.assertNotIn(["sudo", "-v"], argvs)
        self.assertFalse(any(c and c[0] == "killall" for c in argvs))
        self.assertFalse(target.exists())

    def test_dotfiles_sync_creates_archive_and_copies(self):
        (self.repo / ".zshrc").write_text("export FOO=1")
        runner = FakeRunner()

        code = self._main(["-d", "-f"], runner)

        self.assertEqual(code, 0)
        self.assertEqual((self.home / ".zshrc").read_text(), "export FOO=1")
        self.assertTrue((self.home / ".dotfile-archive" / "latest").exists())

    def test_install_logs_archive_path_at_info(self):
        (self.repo / ".zshrc").write_text("export FOO=1")
        runner = FakeRunner()

        with self.assertLogs("macos_setup.__main__", level="INFO") as cm:
            self._main(["-d", "-f"], runner)
        self.assertTrue(any("Archive" in line for line in cm.output))

    def test_uninstall_logs_reverted_from_at_info(self):
        root = self.home / ".dotfile-archive"
        archive = Archive.create(root, "ts1")
        target = self.home / ".zshrc"
        target.write_text("managed")
        archive.record_file(str(target), "added", hashlib.sha256(b"managed").hexdigest())
        archive.save()
        runner = FakeRunner()

        with self.assertLogs("macos_setup.__main__", level="INFO") as cm:
            self._main(["--uninstall", "-f"], runner)
        self.assertTrue(any("Reverted" in line for line in cm.output))



def _admin_mac(argv):
    """A Mac where Homebrew is installed and writable and macOS reports a version."""
    if argv == ["sw_vers", "-productVersion"]:
        return CompletedResult(0, "15.0\n", "")
    if argv == ["brew", "--version"]:
        return CompletedResult(0, "Homebrew 4.0.0", "")
    if argv in (["brew", "--repository"], ["brew", "--prefix"]):
        return CompletedResult(0, "/opt/homebrew\n", "")
    return None


class PrivilegeOrderTests(unittest.TestCase):
    """Per-user work runs first, Homebrew (admin group) next, and the sudo prompt last."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        env = mock.patch.dict(os.environ, {"PATH": "/usr/bin"})
        env.start()
        self.addCleanup(env.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _main(self, argv, runner):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(argv, runner=runner, home=self.home, repo=self.repo)

    @staticmethod
    def _index(calls, predicate, *, last=False):
        hits = [i for i, call in enumerate(calls) if predicate(call)]
        assert hits, "no matching call"
        return hits[-1] if last else hits[0]

    def test_user_work_then_brew_then_sudo(self):
        runner = FakeRunner(_admin_mac)

        code = self._main(["-bmr", "-f"], runner)

        calls = runner.calls
        self.assertEqual(code, 0)
        sudo_prompt = self._index(calls, lambda c: c["argv"] == ["sudo", "-v"])
        first_brew = self._index(calls, lambda c: c["argv"][:2] == ["brew", "update"])
        last_brew = self._index(calls, lambda c: c["argv"] == ["brew", "cleanup"])
        last_rustup = self._index(calls, lambda c: c["argv"][0] == "rustup", last=True)
        last_user_write = self._index(
            calls,
            lambda c: c["argv"][:2] == ["defaults", "write"] and not c["sudo"],
            last=True,
        )
        library = self._index(
            calls, lambda c: c["argv"] == ["chflags", "nohidden", str(self.home / "Library")]
        )
        first_sudo = self._index(calls, lambda c: c["sudo"])
        for user_step in (last_rustup, last_user_write, library):
            self.assertLess(user_step, first_brew)
        self.assertLess(last_brew, sudo_prompt)
        self.assertLess(sudo_prompt, first_sudo)

    def test_steps_without_sudo_work_never_prompt(self):
        for flags in (["-d"], ["-b"], ["-r"]):
            with self.subTest(flags=flags):
                runner = FakeRunner(_admin_mac)
                self._main([*flags, "-f"], runner)
                self.assertNotIn(["sudo", "-v"], runner.argv_list())

    def test_standard_user_skips_brew_but_keeps_user_tooling(self):
        def handler(argv):
            if argv[:2] == ["test", "-w"]:
                return CompletedResult(1, "", "")
            return _admin_mac(argv)

        runner = FakeRunner(handler)

        code = self._main(["-b", "-f"], runner)

        argvs = runner.argv_list()
        self.assertEqual(code, 0)
        self.assertIn(["rustup", "default", "stable"], argvs)
        self.assertNotIn(["brew", "install", "coreutils"], argvs)

    def test_declined_sudo_keeps_per_user_changes(self):
        def handler(argv):
            if argv == ["sudo", "-v"]:
                return CompletedResult(1, "", "Sorry, try again.")
            return _admin_mac(argv)

        runner = FakeRunner(handler)

        with self.assertLogs("macos_setup.__main__", level="WARNING") as cm:
            code = self._main(["-mr", "-f"], runner)

        self.assertEqual(code, 0)
        self.assertTrue(any("Skipped system-wide" in line for line in cm.output))
        self.assertFalse(any(call["sudo"] for call in runner.calls))
        self.assertTrue(any(argv and argv[0] == "killall" for argv in runner.argv_list()))
        latest = Archive.load(self.home / ".dotfile-archive" / "latest")
        scopes = {record["scope"] for record in latest.manifest["settings"]}
        self.assertTrue(scopes)
        self.assertNotIn("sudo", scopes)

    def test_cancelled_sudo_prompt_keeps_per_user_changes(self):
        def handler(argv):
            if argv == ["sudo", "-v"]:
                raise KeyboardInterrupt
            return _admin_mac(argv)

        runner = FakeRunner(handler)

        code = self._main(["-m", "-f"], runner)

        self.assertEqual(code, 0)
        self.assertFalse(any(call["sudo"] for call in runner.calls))

    def test_dry_run_never_prompts_for_sudo(self):
        runner = FakeRunner(_admin_mac)
        self._main(["-bm", "--dry-run", "-f"], runner)
        self.assertNotIn(["sudo", "-v"], runner.argv_list())

    def test_uninstall_reverts_user_scope_before_sudo(self):
        root = self.home / ".dotfile-archive"
        archive = Archive.create(root, "ts1")
        archive.record_setting("user", "com.apple.dock", "tilesize", present=False, applied=36)
        archive.record_setting("sudo", "com.apple.Safari", "HomePage", present=False, applied="x")
        archive.save()

        def handler(argv):
            if argv[:2] == ["defaults", "export"]:
                domain = argv[2]
                value = {"com.apple.dock": "<integer>36</integer>", "com.apple.Safari": "<string>x</string>"}[domain]
                key = "tilesize" if domain == "com.apple.dock" else "HomePage"
                return CompletedResult(
                    0, f'<?xml version="1.0"?><plist version="1.0"><dict><key>{key}</key>{value}</dict></plist>', ""
                )
            return None

        runner = FakeRunner(handler)

        code = self._main(["--uninstall", "-f"], runner)

        argvs = runner.argv_list()
        self.assertEqual(code, 0)
        user_delete = argvs.index(["defaults", "delete", "com.apple.dock", "tilesize"])
        self.assertLess(user_delete, argvs.index(["sudo", "-v"]))
        self.assertLess(argvs.index(["sudo", "-v"]), argvs.index(["defaults", "delete", "com.apple.Safari", "HomePage"]))


if __name__ == "__main__":
    unittest.main()
