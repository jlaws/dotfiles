"""Command-line entry point: parse flags, then install or uninstall.

``setup.sh`` execs ``python3 -m macos_setup``. Install steps archive what they change; ``--uninstall``
reverts a run (guarded). ``--dry-run`` previews without writing. See ``README.md`` for usage.

Work is ordered by privilege: per-user changes run and are archived first, Homebrew (admin group)
next, and the single ``sudo`` prompt last. A standard user, or one who cancels the password
prompt, still ends with every per-user change applied.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from macos_setup.archive import Archive, resolve_archive
from macos_setup.brew import install_packages, install_user_tooling
from macos_setup.dotfiles import revert_files, sync_agents, sync_dotfiles
from macos_setup.logging_setup import configure_logging
from macos_setup.macos_defaults import SETTINGS, apply_defaults, revert_defaults, split_by_privilege
from macos_setup.shell import Runner
from macos_setup.system import (
    apply_system,
    apply_user_system,
    has_system_wide,
    revert_system,
    revert_user_system,
)

# Not `logging.getLogger(__name__)`: when this file runs as the entry point (`python3 -m
# macos_setup`), Python sets `__name__` to the literal string "__main__", which would create an
# unrelated top-level logger outside the "macos_setup" hierarchy that configure_logging() never
# attaches a handler to.
_LOG = logging.getLogger("macos_setup.__main__")
_LATEST = "__latest__"

_RESTART_APPS = [
    "Activity Monitor",
    "cfprefsd",
    "Dock",
    "Finder",
    "Google Chrome",
    "Mail",
    "Messages",
    "Photos",
    "Safari",
    "SystemUIServer",
    "Terminal",
]


@dataclass
class Actions:
    """Resolved plan of what a run should do."""

    dotfiles: bool
    agents: bool
    brew: bool
    macos: bool
    restart: bool
    uninstall: bool
    uninstall_ref: str | None
    list_archives: bool
    dry_run: bool
    force: bool
    project: str | None
    verbose: bool

    @property
    def run_all(self) -> bool:
        """Whether every install step is selected (the no-flags default)."""
        return all([self.dotfiles, self.agents, self.brew, self.macos, self.restart])


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (short flags group, e.g. ``-cb``)."""
    parser = argparse.ArgumentParser(
        prog="setup.sh",
        description="Sync dotfiles, install packages, configure macOS, or reverse a run.",
    )
    parser.add_argument("-d", "--dotfiles", action="store_true", help="Sync dotfiles to home")
    parser.add_argument("-c", "--config", action="store_true", help="Sync agent configuration")
    parser.add_argument("-b", "--brew", action="store_true", help="Install Homebrew packages")
    parser.add_argument("-m", "--macos", action="store_true", help="Configure macOS preferences")
    parser.add_argument("-r", "--restart", action="store_true", help="Restart affected apps")
    parser.add_argument("-p", "--project", default=None, help="Target project dir (requires -c)")
    parser.add_argument("-f", "--force", action="store_true", help="Skip confirmation prompts")
    parser.add_argument(
        "--uninstall",
        nargs="?",
        const=_LATEST,
        default=None,
        metavar="TIMESTAMP",
        help="Revert a run (default: the most recent)",
    )
    parser.add_argument(
        "--list-archives", action="store_true", dest="list_archives", help="List archived runs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run", help="Preview changes; write nothing"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show DEBUG-level detail (every command run)"
    )
    return parser


def resolve_actions(ns: argparse.Namespace) -> Actions:
    """Turn parsed flags into an :class:`Actions` plan, validating combinations.

    Raises:
        ValueError: If ``--project`` is used without ``-c``, or ``--uninstall`` with install steps.
    """
    if ns.project and not ns.config:
        raise ValueError("--project/-p requires -c/--config")
    uninstall = ns.uninstall is not None
    steps = any([ns.dotfiles, ns.config, ns.brew, ns.macos, ns.restart])
    if uninstall and steps:
        raise ValueError("--uninstall cannot be combined with install steps")
    run_all = not steps and not uninstall and not ns.list_archives
    ref = None if ns.uninstall in (None, _LATEST) else ns.uninstall
    return Actions(
        dotfiles=ns.dotfiles or run_all,
        agents=ns.config or run_all,
        brew=ns.brew or run_all,
        macos=ns.macos or run_all,
        restart=ns.restart or run_all,
        uninstall=uninstall,
        uninstall_ref=ref,
        list_archives=ns.list_archives,
        dry_run=ns.dry_run,
        force=ns.force,
        project=ns.project,
        verbose=ns.verbose,
    )


def _timestamp() -> str:
    """Return a filesystem-safe archive timestamp."""
    return datetime.now().strftime("%Y-%m-%d-%H%M%S")


def _macos_version(runner: Runner) -> str:
    """Return the macOS product version, or ``0.0`` when unavailable."""
    result = runner.run(["sw_vers", "-productVersion"], capture=True, check=False)
    return result.stdout.strip() or "0.0"


def _confirm(message: str) -> bool:
    """Prompt for confirmation; return True only on an affirmative answer."""
    try:
        return input(message).strip().lower().startswith("y")
    except EOFError:
        return False


def _start_sudo_keepalive(runner: Runner) -> None:
    """Refresh the sudo timestamp in the background so long runs don't re-prompt."""

    def keepalive() -> None:
        while True:
            runner.run(["sudo", "-n", "true"], check=False)
            time.sleep(60)

    thread = threading.Thread(target=keepalive, daemon=True)
    thread.start()


_SUDO_SKIPPED = "Skipped system-wide settings (no administrator privileges); per-user changes applied."


def _close_system_settings(runner: Runner) -> None:
    """Quit System Settings so it does not overwrite the preferences being changed."""
    runner.run(
        ["osascript", "-e", 'tell application "System Settings" to quit'], check=False
    )


def _acquire_sudo(runner: Runner) -> bool:
    """Ask for administrator privileges; return False when denied, failed, or cancelled."""
    _LOG.info("System-wide settings need administrator privileges.")
    try:
        ok = runner.run(["sudo", "-v"], check=False).ok
    except KeyboardInterrupt:
        ok = False
    if ok:
        _start_sudo_keepalive(runner)
    return ok


def restart_apps(runner: Runner, *, dry_run: bool = False) -> None:
    """Restart the apps affected by preference changes."""
    for app in _RESTART_APPS:
        if dry_run:
            _LOG.info("would restart %s", app)
            continue
        runner.run(["killall", app], check=False)
        _LOG.info("restarted %s", app)


def _list_archives(root: Path) -> None:
    """Print archived runs, newest last."""
    entries = (
        sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "latest")
        if root.exists()
        else []
    )
    if not entries:
        print("No archives found.")
        return
    print(f"Archives in {root}:")
    for name in entries:
        print(f"  {name}")


def _uninstall(archive_root: Path, actions: Actions, runner: Runner, home: Path) -> int:
    """Revert a previously archived run."""
    try:
        path = resolve_archive(archive_root, actions.uninstall_ref)
    except FileNotFoundError as exc:
        _LOG.error(str(exc))
        return 1
    if not actions.force and not actions.dry_run and not _confirm(
        f"Revert changes from {path.name}? [y/N] "
    ):
        _LOG.warning("Aborted.")
        return 1
    archive = Archive.load(path)
    touched_macos = bool(archive.manifest.get("settings") or archive.manifest.get("system"))
    if touched_macos and not actions.dry_run:
        _close_system_settings(runner)
    revert_files(archive, dry_run=actions.dry_run)
    revert_defaults(archive, runner, dry_run=actions.dry_run, sudo=False)
    revert_user_system(archive, runner, home, dry_run=actions.dry_run)
    if has_system_wide(archive):
        if actions.dry_run or _acquire_sudo(runner):
            revert_defaults(archive, runner, dry_run=actions.dry_run, sudo=True)
            revert_system(archive, runner, dry_run=actions.dry_run)
        else:
            _LOG.warning(_SUDO_SKIPPED)
    if touched_macos and not actions.dry_run:
        restart_apps(runner)
    _LOG.info("Reverted from %s", path)
    return 0


def _install(actions: Actions, runner: Runner, repo: Path, home: Path, archive_root: Path) -> int:
    """Run the selected install steps, archiving everything they change."""
    if actions.run_all and not actions.force and not actions.dry_run and not _confirm(
        "This will overwrite files and change system settings. Continue? [y/N] "
    ):
        _LOG.warning("Aborted.")
        return 1
    target = Path(actions.project) if actions.project else home
    if actions.dry_run:
        archive = Archive.in_memory(_timestamp())
    else:
        archive = Archive.create(archive_root, _timestamp())
        archive.manifest["macos_version"] = _macos_version(runner)

    per_user, system_wide = split_by_privilege(SETTINGS)
    version = _macos_version(runner) if actions.macos else ""

    # 1. Per-user changes: no admin group, no sudo.
    if actions.dotfiles:
        sync_dotfiles(repo, home, archive, dry_run=actions.dry_run)
    if actions.agents:
        sync_agents(repo, target, archive, dry_run=actions.dry_run)
    if actions.brew:
        install_user_tooling(runner, dry_run=actions.dry_run)
    if actions.macos:
        if not actions.dry_run:
            _close_system_settings(runner)
        apply_defaults(per_user, archive, runner, version, dry_run=actions.dry_run)
        apply_user_system(archive, runner, home, dry_run=actions.dry_run)
    if not actions.dry_run:
        archive.save()

    # 2. Homebrew: needs a writable prefix (admin group); skipped with a warning otherwise.
    if actions.brew:
        install_packages(runner, dry_run=actions.dry_run)

    # 3. System-wide changes: the one sudo prompt, last, so cancelling it loses nothing above.
    if actions.macos:
        if actions.dry_run or _acquire_sudo(runner):
            apply_defaults(system_wide, archive, runner, version, dry_run=actions.dry_run)
            apply_system(archive, runner, home, dry_run=actions.dry_run)
        else:
            _LOG.warning(_SUDO_SKIPPED)

    if actions.restart:
        restart_apps(runner, dry_run=actions.dry_run)
    if not actions.dry_run:
        archive.save()
        _LOG.info("Archive: %s", archive.path)
    return 0


def main(
    argv: list[str] | None = None,
    *,
    runner: Runner | None = None,
    home: Path | None = None,
    repo: Path | None = None,
) -> int:
    """Parse arguments and dispatch to list/uninstall/install."""
    ns = build_parser().parse_args(argv)
    configure_logging(verbose=ns.verbose)
    try:
        actions = resolve_actions(ns)
    except ValueError as exc:
        _LOG.error(str(exc))
        return 2

    runner = runner or Runner()
    home = home or Path.home()
    repo = repo or Path(__file__).resolve().parent.parent
    archive_root = home / ".dotfile-archive"

    if actions.list_archives:
        _list_archives(archive_root)
        return 0
    if actions.uninstall:
        return _uninstall(archive_root, actions, runner, home)
    return _install(actions, runner, repo, home, archive_root)


if __name__ == "__main__":
    raise SystemExit(main())
