"""Tests for .extra environment and file descriptor limits."""

import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRA_PATH = REPO_ROOT / ".extra"
ZSHRC_PATH = REPO_ROOT / ".zshrc"


class ExtraFileLimitTests(unittest.TestCase):
    def test_zsh_raises_ulimit(self):
        cmd = f"ulimit -Sn 256\nsource {EXTRA_PATH} 2>/dev/null\nulimit -n"
        result = subprocess.run(
            ["zsh", "-c", cmd],
            capture_output=True,
            text=True,
            check=True,
        )
        limit = int(result.stdout.strip())
        self.assertGreaterEqual(limit, 10240)
        self.assertEqual(limit, 65536)

    def test_bash_raises_ulimit(self):
        cmd = f"ulimit -Sn 256\nsource {EXTRA_PATH} 2>/dev/null\nulimit -n"
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            check=True,
        )
        limit = int(result.stdout.strip())
        self.assertGreaterEqual(limit, 10240)
        self.assertEqual(limit, 65536)

    def test_preserves_higher_limit(self):
        cmd = f"ulimit -n 245760 2>/dev/null\nsource {EXTRA_PATH} 2>/dev/null\nulimit -n"
        result = subprocess.run(
            ["zsh", "-c", cmd],
            capture_output=True,
            text=True,
            check=True,
        )
        limit = int(result.stdout.strip())
        self.assertEqual(limit, 245760)

    def test_handles_unlimited_without_error(self):
        cmd = (
            "current_limit=unlimited\n"
            'if [ "$current_limit" != "unlimited" ] && [ "${current_limit:-0}" -lt 65536 ] 2>/dev/null; then\n'
            "    ulimit -n 65536 2>/dev/null || ulimit -n 10240 2>/dev/null\n"
            "fi\n"
            "echo OK"
        )
        result = subprocess.run(
            ["zsh", "-c", cmd],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "OK")


class ShellStartupTests(unittest.TestCase):
    """`.zshrc` and `.extra` must work for any account, prefix, and fresh machine."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name) / "home"
        self.home.mkdir()
        (self.home / ".extra").write_text(EXTRA_PATH.read_text())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _zsh(self, script: str, **env: str) -> subprocess.CompletedProcess[str]:
        """Source `.zshrc` as a new shell would, from a home with nothing else installed.

        `DOTFILES_BREW_CANDIDATES` points at a missing path by default, so the host's own
        Homebrew never feeds `brew shellenv` into a test.
        """
        base = {
            "HOME": str(self.home),
            "PATH": "/usr/bin:/bin",
            "TERM": "dumb",
            "DOTFILES_BREW_CANDIDATES": str(Path(self._tmp.name) / "no-brew"),
        }
        return subprocess.run(
            ["zsh", "-f", "-c", f"source {shlex.quote(str(ZSHRC_PATH))}\n{script}"],
            capture_output=True,
            text=True,
            check=True,
            env={**base, **env},
        )

    def _prefix(self) -> Path:
        """A fake Homebrew prefix with completions, openssl, and autojump."""
        prefix = Path(self._tmp.name) / "brew"
        (prefix / "share" / "zsh" / "site-functions").mkdir(parents=True)
        (prefix / "opt" / "openssl" / "bin").mkdir(parents=True)
        (prefix / "etc" / "profile.d").mkdir(parents=True)
        (prefix / "etc" / "profile.d" / "autojump.sh").write_text("AUTOJUMP_LOADED=1\n")
        return prefix

    def _paths(self, **env: str) -> tuple[list[str], list[str]]:
        result = self._zsh('print -rl -- "${path[@]}"; print -- ---; print -rl -- "${fpath[@]}"', **env)
        path, fpath = result.stdout.split("---\n")
        return path.splitlines(), fpath.splitlines()

    def test_fresh_home_starts_without_errors(self):
        """No oh-my-zsh, Homebrew, or `~/.work` (all optional) must not print errors."""
        result = self._zsh("true")
        self.assertEqual(result.stderr, "")

    def test_missing_extra_starts_without_errors(self):
        (self.home / ".extra").unlink()
        result = self._zsh("true")
        self.assertEqual(result.stderr, "")

    def test_path_names_no_other_account_or_unresolved_prefix(self):
        path, _ = self._paths()
        self.assertFalse([e for e in path if e.startswith("/Users/") and not e.startswith(str(self.home))])
        self.assertNotIn("/opt/openssl/bin", path)
        self.assertNotIn("/opt/homebrew/opt/rustup/bin", path)

    def test_homebrew_paths_follow_the_prefix(self):
        prefix = self._prefix()
        path, fpath = self._paths(HOMEBREW_PREFIX=str(prefix))
        self.assertIn(str(prefix / "opt" / "openssl" / "bin"), path)
        self.assertIn(str(prefix / "share" / "zsh" / "site-functions"), fpath)
        self.assertNotIn("/opt/homebrew/share/zsh/site-functions", fpath)

    def test_autojump_loads_from_the_prefix(self):
        result = self._zsh('print -r -- "${AUTOJUMP_LOADED:-}"', HOMEBREW_PREFIX=str(self._prefix()))
        self.assertEqual(result.stdout.strip(), "1")

    def test_prefix_without_openssl_adds_no_openssl_entry(self):
        prefix = Path(self._tmp.name) / "bare-brew"
        prefix.mkdir()
        path, _ = self._paths(HOMEBREW_PREFIX=str(prefix))
        self.assertFalse([e for e in path if "openssl" in e])

    def test_unset_prefix_falls_back_to_first_installed_brew(self):
        prefix = self._prefix()
        fake_brew = Path(self._tmp.name) / "bin" / "brew"
        fake_brew.parent.mkdir()
        fake_brew.write_text(f"#!/bin/sh\necho 'export HOMEBREW_PREFIX={prefix}'\n")
        fake_brew.chmod(0o755)
        missing = Path(self._tmp.name) / "missing" / "brew"
        path, fpath = self._paths(DOTFILES_BREW_CANDIDATES=f"{missing} {fake_brew}")
        self.assertIn(str(prefix / "opt" / "openssl" / "bin"), path)
        self.assertIn(str(prefix / "share" / "zsh" / "site-functions"), fpath)

    def test_oh_my_zsh_sourced_when_installed(self):
        omz = self.home / ".oh-my-zsh"
        omz.mkdir()
        (omz / "oh-my-zsh.sh").write_text("OMZ_LOADED=1\n")
        result = self._zsh('print -r -- "${OMZ_LOADED:-}"')
        self.assertEqual(result.stdout.strip(), "1")

    def test_antigravity_bin_comes_from_home(self):
        bin_dir = self.home / ".antigravity" / "antigravity" / "bin"
        bin_dir.mkdir(parents=True)
        path, _ = self._paths()
        self.assertIn(str(bin_dir), path)

    def test_missing_antigravity_bin_stays_off_path(self):
        path, _ = self._paths()
        self.assertFalse([e for e in path if "antigravity" in e])


if __name__ == "__main__":
    unittest.main()
