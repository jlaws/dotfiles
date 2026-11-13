"""Tests for .extra environment and file descriptor limits."""

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
        """Source `.zshrc` as a new shell would, from a home with nothing else installed."""
        base = {"HOME": str(self.home), "PATH": "/usr/bin:/bin", "TERM": "dumb"}
        return subprocess.run(
            ["zsh", "-f", "-c", f"source {ZSHRC_PATH}\n{script}"],
            capture_output=True,
            text=True,
            check=True,
            env={**base, **env},
        )

    def test_fresh_home_starts_without_errors(self):
        """No oh-my-zsh and no `~/.work` (both optional) must not print errors on every shell."""
        result = self._zsh("true")
        self.assertEqual(result.stderr, "")

    def test_path_names_no_other_account_or_unresolved_prefix(self):
        result = self._zsh('print -rl -- "${path[@]}"')
        entries = result.stdout.splitlines()
        self.assertFalse(
            [e for e in entries if e.startswith("/Users/") and not e.startswith(str(self.home))]
        )
        self.assertNotIn("/opt/openssl/bin", entries)

    def test_homebrew_paths_follow_the_prefix(self):
        prefix = Path(self._tmp.name) / "brew"
        (prefix / "share" / "zsh" / "site-functions").mkdir(parents=True)
        (prefix / "opt" / "openssl" / "bin").mkdir(parents=True)
        result = self._zsh(
            'print -rl -- "${path[@]}"; print -- ---; print -rl -- "${fpath[@]}"',
            HOMEBREW_PREFIX=str(prefix),
        )
        path, fpath = result.stdout.split("---\n")
        self.assertIn(str(prefix / "opt" / "openssl" / "bin"), path.splitlines())
        self.assertIn(str(prefix / "share" / "zsh" / "site-functions"), fpath.splitlines())
        self.assertNotIn("/opt/homebrew/share/zsh/site-functions", fpath.splitlines())

    def test_antigravity_bin_comes_from_home(self):
        bin_dir = self.home / ".antigravity" / "antigravity" / "bin"
        bin_dir.mkdir(parents=True)
        result = self._zsh('print -rl -- "${path[@]}"')
        self.assertIn(str(bin_dir), result.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()
