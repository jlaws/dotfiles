"""Install Homebrew packages and language tooling (mirrors the original install step).

Package installs are not reversed by ``--uninstall``: removing tools the user may now depend on
(git, node, vim) is riskier than leaving them. This module only applies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from macos_setup.shell import Runner

_LOG = logging.getLogger(__name__)

# Homebrew formulae, in install order. coreutils is first (it gets a follow-up symlink).
BREW_PACKAGES = [
    "coreutils",
    "moreutils",
    "findutils",
    "fd",
    "wget",
    "just",
    "vim",
    "grep",
    "openssh",
    "git",
    "gh",
    "autojump",
    "mermaid-cli",
    "poppler",
    "uv",
    "node",
]

_ELAN_INSTALL = (
    "curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh "
    "| sh -s -- -y --default-toolchain none"
)
_CLAUDE_INSTALL = "curl -fsSL https://claude.ai/install.sh | bash"
_RUST_INSTALL = "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"


def _run(runner: Runner, argv: list[str], *, dry_run: bool, check: bool = True) -> None:
    """Run or, in dry-run mode, log a single install command."""
    if dry_run:
        _LOG.info("would run: %s", " ".join(argv))
        return
    _LOG.info("run: %s", " ".join(argv))
    runner.run(argv, check=check)


def install_packages(runner: Runner, *, dry_run: bool = False) -> None:
    """Update Homebrew, install packages and language servers, then clean up.

    Raises:
        RuntimeError: If Homebrew is not installed.
    """
    if not runner.run(["brew", "--version"], capture=True, check=False).ok:
        message = "Homebrew is required but not installed. Install it from https://brew.sh first."
        _LOG.error(message)
        raise RuntimeError(message)

    _run(runner, ["brew", "update"], dry_run=dry_run)
    _run(runner, ["brew", "upgrade"], dry_run=dry_run)

    for package in BREW_PACKAGES:
        _run(runner, ["brew", "install", package], dry_run=dry_run)

    if not dry_run:
        prefix = runner.run(["brew", "--prefix"], capture=True, check=False).stdout.strip()
        if prefix:
            _run(
                runner,
                ["ln", "-sf", f"{prefix}/bin/gsha256sum", f"{prefix}/bin/sha256sum"],
                dry_run=dry_run,
                check=False,
            )

    _run(runner, ["bash", "-c", _RUST_INSTALL], dry_run=dry_run)
    _run(runner, ["rustup", "default", "stable"], dry_run=dry_run)
    _run(runner, ["rustup", "component", "add", "rust-analyzer"], dry_run=dry_run)
    _run(runner, ["npm", "install", "-g", "typescript-language-server", "typescript"], dry_run=dry_run)
    _run(runner, ["bash", "-c", _ELAN_INSTALL], dry_run=dry_run)
    _run(runner, ["bash", "-c", _CLAUDE_INSTALL], dry_run=dry_run)
    _run(runner, ["brew", "cleanup"], dry_run=dry_run)
