"""End-to-end test that `python3 -m macos_setup` actually logs to stdout.

Regression coverage for a real bug: modules that call `logging.getLogger(__name__)` get a logger
named the literal string "__main__" when run via `python3 -m macos_setup` (Python's own rule for
the entry-point module), not "macos_setup.__main__". That logger falls outside the configured
package hierarchy, so its messages silently vanish -- a mistake unit tests that import the module
normally (where `__name__` is already the qualified path) cannot catch. Only running the real
`-m` invocation as a subprocess exercises this.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class CliInvocationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dotfiles_sync_logs_archive_path_when_invoked_as_module(self):
        (REPO_ROOT / ".zshrc").exists()  # sanity: repo has at least one syncable dotfile
        result = subprocess.run(
            [sys.executable, "-m", "macos_setup", "-d", "-f"],
            cwd=REPO_ROOT,
            env={"HOME": str(self.tmp), "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("INFO", result.stdout)
        self.assertIn("Archive:", result.stdout)


if __name__ == "__main__":
    unittest.main()
