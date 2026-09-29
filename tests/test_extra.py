"""Tests for .extra environment and file descriptor limits."""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRA_PATH = REPO_ROOT / ".extra"


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


if __name__ == "__main__":
    unittest.main()
