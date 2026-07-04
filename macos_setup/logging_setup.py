"""Console logging configuration for the macos_setup CLI.

Library code only attaches a :class:`logging.NullHandler` (see ``macos_setup/__init__.py``), so
importing the package is silent. The CLI entry point calls :func:`configure_logging` once at
startup to attach a stdout handler with timestamps and levels: INFO narrates progress, WARNING
flags guard-skips and best-effort limitations, DEBUG (via ``--verbose``) shows every command run.
"""

from __future__ import annotations

import logging
import sys
from typing import IO

PACKAGE_LOGGER = "macos_setup"


def configure_logging(verbose: bool = False, stream: IO[str] | None = None) -> None:
    """Attach a stream handler to the package logger, replacing any previously attached one.

    Rebinding on every call (rather than a one-time "already configured" guard) keeps this correct
    under ``contextlib.redirect_stdout`` in tests, where ``sys.stdout`` changes between calls.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]:
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
