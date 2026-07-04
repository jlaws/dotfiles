"""Timestamped archive of everything a run changes, plus the manifest that drives uninstall.

Each run lives in ``<root>/<timestamp>/`` with a ``files/`` tree (replaced originals), a
``domains/`` tree (pre-run ``defaults export`` snapshots), and ``manifest.json``. A ``latest``
symlink in ``root`` points at the newest run so ``--uninstall`` can find it without an argument.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
LATEST = "latest"


def resolve_archive(root: Path, ref: str | None) -> Path:
    """Return the archive directory for ``ref``, or the latest run when ``ref`` is ``None``.

    Raises:
        FileNotFoundError: If no archive matches, or the target lacks a manifest.
    """
    if ref:
        target = root / ref
    else:
        latest = root / LATEST
        if not latest.exists():
            raise FileNotFoundError(f"No archives found in {root}")
        target = latest.resolve()
    if not (target / MANIFEST_NAME).exists():
        raise FileNotFoundError(f"Not a valid archive: {target}")
    return target


def _empty_manifest(timestamp: str) -> dict[str, Any]:
    """Return a fresh, empty manifest dict."""
    return {
        "version": MANIFEST_VERSION,
        "timestamp": timestamp,
        "steps": [],
        "macos_version": "",
        "files": [],
        "settings": [],
        "system": [],
    }


def _update_latest(root: Path, target: Path) -> None:
    """Point ``root/latest`` at ``target``, replacing any existing symlink."""
    link = root / LATEST
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


@dataclass
class Archive:
    """A single run's archive: its directory and in-memory manifest."""

    path: Path
    manifest: dict[str, Any]

    @classmethod
    def create(cls, root: Path, timestamp: str) -> Archive:
        """Create the archive directory tree, write an empty manifest, and update ``latest``."""
        path = root / timestamp
        (path / "files").mkdir(parents=True, exist_ok=True)
        (path / "domains").mkdir(parents=True, exist_ok=True)
        archive = cls(path, _empty_manifest(timestamp))
        archive.save()
        _update_latest(root, path)
        return archive

    @classmethod
    def in_memory(cls, timestamp: str) -> Archive:
        """Return an archive backed by no directory, for dry runs that must not write."""
        return cls(Path("<dry-run>"), _empty_manifest(timestamp))

    @classmethod
    def load(cls, path: Path) -> Archive:
        """Load an existing archive's manifest from disk."""
        manifest = json.loads((path / MANIFEST_NAME).read_text())
        return cls(path, manifest)

    @property
    def files_dir(self) -> Path:
        """Directory holding originals of replaced files."""
        return self.path / "files"

    @property
    def domains_dir(self) -> Path:
        """Directory holding pre-run preference-domain snapshots."""
        return self.path / "domains"

    def record_file(self, dest: str, action: str, sha256: str) -> None:
        """Record a synced file: ``action`` is ``"replaced"`` or ``"added"``."""
        self.manifest["files"].append({"dest": dest, "action": action, "sha256": sha256})

    def record_setting(
        self, scope: str, domain: str, key: str, *, present: bool, applied: Any
    ) -> None:
        """Record a changed macOS default and the value setup applied."""
        self.manifest["settings"].append(
            {
                "scope": scope,
                "domain": domain,
                "key": key,
                "present": present,
                "applied": applied,
            }
        )

    def record_system(self, name: str, *, present: bool, original: Any, applied: Any) -> None:
        """Record a changed system-level setting (pmset/nvram/systemsetup/firewall).

        ``original`` is the captured prior value (for restore); ``applied`` is what setup set
        (for the revert guard). ``present`` is whether the value existed before setup ran.
        """
        self.manifest["system"].append(
            {"name": name, "present": present, "original": original, "applied": applied}
        )

    def save(self) -> None:
        """Write the manifest to disk."""
        (self.path / MANIFEST_NAME).write_text(json.dumps(self.manifest, indent=2) + os.linesep)
