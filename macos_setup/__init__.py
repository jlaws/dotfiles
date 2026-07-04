"""macOS setup: sync dotfiles, install packages, configure macOS, and reverse it all.

`setup.sh` is a shim that runs this package. Every run archives what it changes so a later
``--uninstall`` can restore the prior state (guarded: only reverts values it still owns).
"""

import logging

# Library-safe default: emit nothing until the CLI entry point calls
# `macos_setup.logging_setup.configure_logging()`. `propagate = False` stops records from
# reaching the root logger's handler-of-last-resort, which would otherwise print WARNING+ to
# stderr even when nothing has been configured.
_package_logger = logging.getLogger(__name__)
_package_logger.addHandler(logging.NullHandler())
_package_logger.propagate = False
