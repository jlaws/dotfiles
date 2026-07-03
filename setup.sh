#!/usr/bin/env bash
# =============================================================================
# macOS Setup - entry point
# =============================================================================
# Thin shim that runs the stdlib-only `macos_setup` Python package. All logic
# (sync dotfiles, install packages, configure macOS, archive, and uninstall)
# lives there. Run `./setup.sh --help` for usage.
#
# Examples:
#   ./setup.sh                # Interactive: run all steps
#   ./setup.sh -cb            # Sync agent config + install Homebrew packages
#   ./setup.sh --list-archives
#   ./setup.sh --uninstall    # Revert the most recent run (guarded)
#   ./setup.sh --dry-run -m   # Preview macOS changes; write nothing
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m macos_setup "$@"
