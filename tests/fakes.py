"""Shared test doubles for the macos_setup suite."""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from macos_setup.shell import CompletedResult, Runner


class FakeRunner(Runner):
    """Records commands and returns canned results instead of shelling out.

    Pass a ``handler`` that maps an argv list to a :class:`CompletedResult` (or ``None`` to
    fall back to a zero-exit empty result). Every call is appended to ``calls``.
    """

    def __init__(self, handler: Callable[[list[str]], CompletedResult | None] | None = None):
        self.handler = handler
        self.calls: list[dict] = []

    def run(
        self,
        argv: list[str],
        *,
        sudo: bool = False,
        capture: bool = False,
        check: bool = True,
    ) -> CompletedResult:
        """Record the call and return the handler's result (or a zero-exit default)."""
        self.calls.append(
            {"argv": list(argv), "sudo": sudo, "capture": capture, "check": check}
        )
        result = self.handler(list(argv)) if self.handler else None
        if result is None:
            result = CompletedResult(0, "", "")
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, argv, result.stdout, result.stderr
            )
        return result

    def argv_list(self) -> list[list[str]]:
        """Return just the argv of each recorded call, in order."""
        return [call["argv"] for call in self.calls]
