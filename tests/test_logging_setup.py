"""Tests for macos_setup.logging_setup."""

import io
import logging
import unittest
from contextlib import redirect_stderr

from macos_setup.logging_setup import PACKAGE_LOGGER, configure_logging


class ConfigureLoggingTests(unittest.TestCase):
    def tearDown(self):
        # Strip only the StreamHandler configure_logging() adds, mirroring its own cleanup.
        # Removing the module-level NullHandler here would make later tests' "found == 0" trip
        # the logging module's handler-of-last-resort regardless of `propagate`.
        logger = logging.getLogger(PACKAGE_LOGGER)
        for handler in [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]:
            logger.removeHandler(handler)

    def test_info_visible_by_default(self):
        stream = io.StringIO()
        configure_logging(stream=stream)
        logging.getLogger(f"{PACKAGE_LOGGER}.x").info("hello")
        self.assertIn("hello", stream.getvalue())

    def test_debug_hidden_unless_verbose(self):
        stream = io.StringIO()
        configure_logging(stream=stream)
        logging.getLogger(f"{PACKAGE_LOGGER}.x").debug("quiet")
        self.assertNotIn("quiet", stream.getvalue())

    def test_debug_visible_when_verbose(self):
        stream = io.StringIO()
        configure_logging(verbose=True, stream=stream)
        logging.getLogger(f"{PACKAGE_LOGGER}.x").debug("loud")
        self.assertIn("loud", stream.getvalue())

    def test_reconfiguring_does_not_duplicate_handlers(self):
        configure_logging(stream=io.StringIO())
        second = io.StringIO()
        configure_logging(stream=second)
        logging.getLogger(f"{PACKAGE_LOGGER}.x").info("once")
        self.assertEqual(second.getvalue().count("once"), 1)

    def test_message_includes_level_name(self):
        stream = io.StringIO()
        configure_logging(stream=stream)
        logging.getLogger(f"{PACKAGE_LOGGER}.x").info("hello")
        self.assertIn("INFO", stream.getvalue())

    def test_unconfigured_child_logger_stays_silent(self):
        # Before configure_logging() is ever called, a warning from a submodule must not leak
        # to stderr via the root logger's handler-of-last-resort.
        captured = io.StringIO()
        with redirect_stderr(captured):
            logging.getLogger(f"{PACKAGE_LOGGER}.x").warning("should not print")
        self.assertEqual(captured.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
