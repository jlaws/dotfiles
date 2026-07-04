"""Subprocess boundary and console helpers.

Every external command runs through :class:`Runner`, the single impure seam in the package.
Keeping it isolated lets the rest of the code stay pure: tests inject a fake with the same
``run`` interface instead of shelling out.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

_LOG = logging.getLogger(__name__)


@dataclass
class CompletedResult:
    """Outcome of a :meth:`Runner.run` call."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """Return True when the command exited zero."""
        return self.returncode == 0


def build_argv(argv: list[str], sudo: bool = False) -> list[str]:
    """Return ``argv`` unchanged, or prefixed with ``sudo`` when requested."""
    return ["sudo", *argv] if sudo else list(argv)


class Runner:
    """Runs external commands. Swap for a fake in tests."""

    def run(
        self,
        argv: list[str],
        *,
        sudo: bool = False,
        capture: bool = False,
        check: bool = True,
    ) -> CompletedResult:
        """Execute ``argv`` and return a :class:`CompletedResult`.

        Args:
            argv: Command and arguments.
            sudo: Prefix the command with ``sudo``.
            capture: Capture stdout/stderr instead of inheriting the terminal.
            check: Raise :class:`subprocess.CalledProcessError` on a nonzero exit.
        """
        full = build_argv(argv, sudo=sudo)
        _LOG.debug("run: %s", " ".join(full))
        proc = subprocess.run(
            full,
            check=False,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
        _LOG.debug("exit %d: %s", proc.returncode, " ".join(full))
        result = CompletedResult(proc.returncode, proc.stdout or "", proc.stderr or "")
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, full, result.stdout, result.stderr
            )
        return result
