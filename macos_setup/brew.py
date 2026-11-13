"""Install Homebrew packages (which need an admin-group account).

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
    "jq",
    "openssh",
    "git",
    "gh",
    "autojump",
    "mermaid-cli",
    "poppler",
    "agent-browser",
    "uv",
    "node",
    "rg",
]


def _run(runner: Runner, argv: list[str], *, dry_run: bool, check: bool = True) -> None:
    """Run or, in dry-run mode, log a single install command."""
    if dry_run:
        _LOG.info("would run: %s", " ".join(argv))
        return
    _LOG.info("run: %s", " ".join(argv))
    runner.run(argv, check=check)


def brew_writable(runner: Runner) -> bool:
    """Return True when Homebrew is installed and this user can write to its repository.

    Homebrew's tree is owned by the installing admin with group ``admin``; a standard user
    can read it but cannot ``brew install``. Checking write access up front lets setup skip
    the Homebrew step with a warning instead of failing partway.
    """
    if not runner.run(["brew", "--version"], capture=True, check=False).ok:
        _LOG.warning(
            "Skipped Homebrew packages: Homebrew is not installed (see https://brew.sh)."
        )
        return False
    repo = runner.run(["brew", "--repository"], capture=True, check=False).stdout.strip()
    if not repo or not runner.run(["test", "-w", repo], check=False).ok:
        _LOG.warning(
            "Skipped Homebrew packages: %s is not writable by this user "
            "(Homebrew needs an account in the admin group).",
            repo or "the Homebrew repository",
        )
        return False
    return True


def install_packages(runner: Runner, *, dry_run: bool = False) -> bool:
    """Update Homebrew, install packages and brew-backed tooling, then clean up.

    Needs an account that can write the Homebrew prefix (the admin group). When Homebrew is
    missing or not writable, logs a warning and returns False without changing anything.
    """
    if not brew_writable(runner):
        return False

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

    # The formula ships only the CLI; the browser it drives is a separate download
    # ("agent-browser install # Download Chrome (first time)" in its own help). Believed
    # idempotent -- the help text says "first time" -- but nothing here pins that, so the claim is
    # stated as belief rather than fact.
    #
    # check=False because this is the one bootstrap step that pulls a large binary over the
    # network, and it sits ahead of the npm language servers and `brew cleanup`. A flaky download
    # should not take those with it. The `ln -sf` above is soft for the same reason.
    _run(runner, ["agent-browser", "install"], dry_run=dry_run, check=False)

    # npm comes from the brew `node` formula, so its global prefix is the Homebrew prefix.
    _run(runner, ["npm", "install", "-g", "typescript-language-server", "typescript"], dry_run=dry_run)
    _run(runner, ["brew", "cleanup"], dry_run=dry_run)
    return True
