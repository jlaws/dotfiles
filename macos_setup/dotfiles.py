"""Sync dotfiles and agent configs, tracking every change for a guarded uninstall.

Applying a file archives the version it replaces (or notes that it was newly added) and records
the sha256 of what setup wrote. Revert restores or deletes a file only when it still matches that
sha, leaving anything edited after setup untouched.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from macos_setup.archive import Archive

_LOG = logging.getLogger(__name__)

# Root dotfiles synced into the home directory (matches the original rsync allowlist).
ROOT_DOTFILES = [
    ".editorconfig",
    ".extra",
    ".gitattributes",
    ".gitconfig",
    ".gitignore",
    ".hushlogin",
    ".python-version",
    ".vimrc",
    ".wgetrc",
    ".zshrc",
]

# Agent-config sync pairs, relative to (repo, target). Files and directories both allowed;
# directories are copied recursively. Cross-tree entries reuse the shared .agents/ knowledge base.
AGENT_SYNC = [
    (".agents/skills", ".agents/skills"),
    (".agents/references", ".agents/references"),
    (".claude/CLAUDE.md", ".claude/CLAUDE.md"),
    (".claude/settings.json", ".claude/settings.json"),
    (".claude/agents", ".claude/agents"),
    (".claude/commands", ".claude/commands"),
    (".claude/hooks", ".claude/hooks"),
    (".claude/skills", ".claude/skills"),
    (".claude/references", ".claude/references"),
    (".cursor/cli-config.json", ".cursor/cli-config.json"),
    (".cursor/hooks.json", ".cursor/hooks.json"),
    (".cursor/hooks", ".cursor/hooks"),
    (".cursor/rules", ".cursor/rules"),
    (".agents/skills", ".cursor/skills"),
    (".agents/references", ".cursor/references"),
    (".codex/AGENTS.md", ".codex/AGENTS.md"),
    (".codex/config.toml", ".codex/config.toml"),
    (".codex/agents", ".codex/agents"),
    (".codex/prompts", ".codex/prompts"),
    (".codex/hooks", ".codex/hooks"),
    (".codex/rules", ".codex/rules"),
    (".gemini/GEMINI.md", ".gemini/GEMINI.md"),
    (".gemini/settings.json", ".gemini/settings.json"),
    (".gemini/policies", ".gemini/policies"),
    (".gemini/agents", ".gemini/agents"),
    (".gemini/commands", ".gemini/commands"),
    (".gemini/hooks", ".gemini/hooks"),
]

# Stale files removed from the target after syncing (archived first so uninstall can restore).
AGENT_REMOVALS = [
    ".claude/hooks/lessons-learned.sh",
    ".cursor/hooks/lessons-learned.sh",
    ".codex/hooks/lessons-learned.sh",
    ".gemini/hooks/lessons-learned.sh",
]


@dataclass
class FileSummary:
    """Destinations restored, deleted, or skipped during a file revert."""

    restored: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_revert_decision(action: str, recorded_sha: str, current_sha: str | None) -> str:
    """Decide how to revert one tracked file: ``restore``, ``delete``, or ``skip``.

    The guard: only act when the file still matches what setup produced. ``replaced`` restores the
    archived original, ``added`` deletes the file, ``removed`` restores a file setup deleted.
    """
    if action == "replaced":
        return "restore" if current_sha == recorded_sha else "skip"
    if action == "added":
        return "delete" if current_sha == recorded_sha else "skip"
    if action == "removed":
        return "restore" if current_sha is None else "skip"
    return "skip"


def _archive_dest(archive: Archive, dest: Path) -> Path:
    """Return the archive path mirroring an absolute destination path."""
    return archive.files_dir / str(dest).lstrip("/")


def apply_file(src: Path, dest: Path, archive: Archive) -> None:
    """Copy ``src`` to ``dest``, archiving any replaced original and recording the change."""
    if dest.exists():
        backup = _archive_dest(archive, dest)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, backup)
        action = "replaced"
    else:
        action = "added"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    archive.record_file(str(dest), action, sha256_file(src))


def remove_file(dest: Path, archive: Archive) -> None:
    """Archive and delete ``dest`` if present; record it so uninstall can restore it."""
    if not dest.exists():
        return
    backup = _archive_dest(archive, dest)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dest, backup)
    archive.record_file(str(dest), "removed", sha256_file(dest))
    dest.unlink()
    _LOG.debug("removed stale file %s", dest)


def revert_files(archive: Archive, *, dry_run: bool = False) -> FileSummary:
    """Restore or delete tracked files whose contents setup still owns; skip user-edited ones."""
    summary = FileSummary()
    for record in archive.manifest.get("files", []):
        dest = Path(record["dest"])
        current = sha256_file(dest) if dest.exists() else None
        decision = file_revert_decision(record["action"], record["sha256"], current)
        if decision == "restore":
            if not dry_run:
                backup = _archive_dest(archive, dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, dest)
            summary.restored.append(str(dest))
            _LOG.info("restore %s", dest)
        elif decision == "delete":
            if not dry_run:
                dest.unlink()
            summary.deleted.append(str(dest))
            _LOG.info("delete %s", dest)
        else:
            summary.skipped.append(str(dest))
            _LOG.warning("skip (user-modified) %s", dest)
    return summary


def _iter_pairs(src: Path, dest: Path) -> Iterator[tuple[Path, Path]]:
    """Yield (src_file, dest_file) pairs for a file or, recursively, a directory."""
    if src.is_dir():
        for child in sorted(src.rglob("*")):
            if child.is_file():
                yield child, dest / child.relative_to(src)
    elif src.exists():
        yield src, dest


def sync_dotfiles(repo: Path, home: Path, archive: Archive, *, dry_run: bool = False) -> None:
    """Sync the root dotfile allowlist from ``repo`` into ``home``."""
    _LOG.info("Syncing %d dotfiles to %s", len(ROOT_DOTFILES), home)
    for name in ROOT_DOTFILES:
        src = repo / name
        if not src.exists():
            continue
        dest = home / name
        if dry_run:
            _LOG.info("would sync %s", dest)
            continue
        apply_file(src, dest, archive)
        _LOG.debug("sync %s", dest)


def sync_agents(repo: Path, target: Path, archive: Archive, *, dry_run: bool = False) -> None:
    """Sync agent configs (Claude, Cursor, Codex, Gemini, shared .agents) into ``target``."""
    _LOG.info("Syncing agent configuration to %s", target)
    synced = 0
    for src_rel, dest_rel in AGENT_SYNC:
        for src_file, dest_file in _iter_pairs(repo / src_rel, target / dest_rel):
            if dry_run:
                _LOG.info("would sync %s", dest_file)
                continue
            apply_file(src_file, dest_file, archive)
            _LOG.debug("sync %s", dest_file)
            synced += 1
    if dry_run:
        return
    for removal in AGENT_REMOVALS:
        remove_file(target / removal, archive)
    _make_hooks_executable(target / ".gemini" / "hooks")
    _LOG.info("Synced %d agent config files to %s", synced, target)


def _make_hooks_executable(hooks_dir: Path) -> None:
    """Mark Gemini hook scripts executable (mirrors the original chmod +x)."""
    if not hooks_dir.is_dir():
        return
    for script in hooks_dir.glob("*.sh"):
        script.chmod(script.stat().st_mode | 0o111)
