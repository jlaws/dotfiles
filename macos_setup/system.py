"""System-level settings (pmset, nvram, systemsetup, firewall, chflags): apply + best-effort revert.

These live outside ``defaults``. Values that read back cleanly (nvram, firewall, systemsetup,
readable pmset keys) are captured and restored under the same guard as everything else. Values
that can't be read back precisely (lidwake, autorestart, standbydelay) are returned to a
documented macOS default on revert and logged.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from macos_setup.archive import Archive
    from macos_setup.shell import Runner

_LOG = logging.getLogger(__name__)

FIREWALL = "/usr/libexec/ApplicationFirewall/socketfilterfw"

# pmset keys set with -a (both power sources) and their applied value.
_PMSET_ALL = [("lidwake", "1"), ("autorestart", "1"), ("hibernatemode", "0")]
# pmset keys set per source: (key, battery, ac).
_PMSET_BC = [
    ("displaysleep", "5", "15"),
    ("sleep", "15", "0"),
    ("standbydelay", "3600", "86400"),
    ("disksleep", "5", "0"),
    ("powernap", "0", "1"),
]
# pmset keys that read back via `pmset -g custom` and can be restored from the snapshot.
_PMSET_READABLE = {"displaysleep", "sleep", "disksleep", "powernap", "hibernatemode"}
# pmset keys that don't read back; returned to a documented default on revert.
_PMSET_DEFAULTS = {"lidwake": "1", "autorestart": "0", "standbydelay": "10800"}


def parse_pmset_custom(text: str) -> dict[str, dict[str, str]]:
    """Parse ``pmset -g custom`` output into ``{"battery": {...}, "ac": {...}}``."""
    result: dict[str, dict[str, str]] = {"battery": {}, "ac": {}}
    section: str | None = None
    for line in text.splitlines():
        if line.startswith("Battery Power"):
            section = "battery"
        elif line.startswith("AC Power"):
            section = "ac"
        elif section and line.strip():
            parts = line.split()
            if len(parts) >= 2:
                result[section][parts[0]] = parts[1]
    return result


def parse_nvram_value(text: str) -> str | None:
    """Return the value from ``nvram <key>`` output, or None when unset (value not stripped)."""
    if not text:
        return None
    line = text.split("\n", 1)[0]
    if "\t" in line:
        return line.split("\t", 1)[1]
    parts = line.split(None, 1)
    return parts[1] if len(parts) == 2 else None


def parse_firewall_state(text: str) -> int | None:
    """Return the integer firewall state from ``--getglobalstate`` output."""
    match = re.search(r"State = (\d+)", text)
    return int(match.group(1)) if match else None


def parse_timezone(text: str) -> str:
    """Return the timezone name from ``systemsetup -gettimezone`` output."""
    return text.partition(":")[2].strip()


def _stealth_state(text: str) -> int:
    """Return 1 when firewall stealth mode is enabled, else 0."""
    return 1 if "enabled" in text.lower() else 0


def _after_colon(text: str) -> str:
    """Return the text after the first colon, stripped."""
    return text.partition(":")[2].strip()


@dataclass
class _Scalar:
    """A single readable system value with apply and restore commands."""

    name: str
    read: list[str]
    read_sudo: bool
    parse: Callable[[str], Any]
    applied: Any
    writes: list[list[str]]
    restore: Callable[[Any, bool], list[list[str]]]


def _scalar_items(home: Path) -> list[_Scalar]:
    """Build the table of readable scalar system settings (shared by apply and revert)."""
    return [
        _Scalar(
            name="nvram:SystemAudioVolume",
            read=["nvram", "SystemAudioVolume"],
            read_sudo=False,
            parse=parse_nvram_value,
            applied=" ",
            writes=[["nvram", "SystemAudioVolume= "]],
            restore=lambda original, present: (
                [["nvram", f"SystemAudioVolume={original}"]]
                if present
                else [["nvram", "-d", "SystemAudioVolume"]]
            ),
        ),
        _Scalar(
            name="firewall:globalstate",
            read=[FIREWALL, "--getglobalstate"],
            read_sudo=False,
            parse=parse_firewall_state,
            applied=1,
            writes=[[FIREWALL, "--setglobalstate", "on"]],
            restore=lambda original, present: [
                [FIREWALL, "--setglobalstate", "on" if original else "off"]
            ],
        ),
        _Scalar(
            name="firewall:stealthmode",
            read=[FIREWALL, "--getstealthmode"],
            read_sudo=False,
            parse=_stealth_state,
            applied=1,
            writes=[[FIREWALL, "--setstealthmode", "on"]],
            restore=lambda original, present: [
                [FIREWALL, "--setstealthmode", "on" if original else "off"]
            ],
        ),
        _Scalar(
            name="systemsetup:timezone",
            read=["systemsetup", "-gettimezone"],
            read_sudo=True,
            parse=parse_timezone,
            applied="America/New_York",
            writes=[["systemsetup", "-settimezone", "America/New_York"]],
            restore=lambda original, present: [["systemsetup", "-settimezone", original]],
        ),
        _Scalar(
            name="systemsetup:computersleep",
            read=["systemsetup", "-getcomputersleep"],
            read_sudo=True,
            parse=_after_colon,
            applied="Off",
            writes=[["systemsetup", "-setcomputersleep", "Off"]],
            restore=lambda original, present: [["systemsetup", "-setcomputersleep", original]],
        ),
        _Scalar(
            name="systemsetup:wakeonnetwork",
            read=["systemsetup", "-getwakeonnetworkaccess"],
            read_sudo=True,
            parse=_after_colon,
            applied="off",
            writes=[["systemsetup", "-setwakeonnetworkaccess", "off"]],
            restore=lambda original, present: [
                ["systemsetup", "-setwakeonnetworkaccess", original or "on"]
            ],
        ),
    ]


def _read(runner: Runner, argv: list[str], sudo: bool) -> str:
    """Return stdout of a read command, or empty string on failure."""
    result = runner.run(argv, sudo=sudo, capture=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def apply_system(archive: Archive, runner: Runner, home: Path, *, dry_run: bool = False) -> None:
    """Capture originals then apply all system-level settings."""
    for item in _scalar_items(home):
        if dry_run:
            _LOG.info("would set %s", item.name)
            continue
        original = item.parse(_read(runner, item.read, item.read_sudo))
        present = original is not None
        archive.record_system(item.name, present=present, original=original, applied=item.applied)
        for write in item.writes:
            runner.run(write, sudo=True)
        _LOG.info("set %s", item.name)

    _apply_pmset(archive, runner, dry_run=dry_run)
    _apply_restartfreeze(archive, runner, dry_run=dry_run)
    _apply_chflags(archive, runner, home, dry_run=dry_run)
    if not dry_run:
        archive.save()


def _apply_pmset(archive: Archive, runner: Runner, *, dry_run: bool) -> None:
    """Snapshot pmset then apply the managed power settings."""
    if dry_run:
        _LOG.info("would configure pmset")
        return
    before = parse_pmset_custom(_read(runner, ["pmset", "-g", "custom"], False))
    applied: dict[str, dict[str, str]] = {"battery": {}, "ac": {}}
    for key, value in _PMSET_ALL:
        applied["battery"][key] = value
        applied["ac"][key] = value
    for key, battery, ac in _PMSET_BC:
        applied["battery"][key] = battery
        applied["ac"][key] = ac
    archive.record_system("pmset", present=True, original=before, applied=applied)
    for key, value in _PMSET_ALL:
        runner.run(["pmset", "-a", key, value], sudo=True)
    for key, battery, ac in _PMSET_BC:
        runner.run(["pmset", "-b", key, battery], sudo=True)
        runner.run(["pmset", "-c", key, ac], sudo=True)
    _LOG.info("configured pmset")


def _apply_restartfreeze(archive: Archive, runner: Runner, *, dry_run: bool) -> None:
    """Enable restart-on-freeze (no getter, so it cannot be precisely restored)."""
    if dry_run:
        _LOG.info("would set systemsetup:restartfreeze")
        return
    archive.record_system("systemsetup:restartfreeze", present=False, original=None, applied="on")
    runner.run(["systemsetup", "-setrestartfreeze", "on"], sudo=True)
    _LOG.info("set systemsetup:restartfreeze")


def _apply_chflags(archive: Archive, runner: Runner, home: Path, *, dry_run: bool) -> None:
    """Reveal ~/Library and /Volumes (default macOS state is hidden)."""
    if dry_run:
        _LOG.info("would reveal ~/Library and /Volumes")
        return
    archive.record_system("chflags:library", present=True, original="hidden", applied="nohidden")
    archive.record_system("chflags:volumes", present=True, original="hidden", applied="nohidden")
    library = str(home / "Library")
    runner.run(["chflags", "nohidden", library], check=False)
    runner.run(["xattr", "-d", "com.apple.FinderInfo", library], check=False)
    runner.run(["chflags", "nohidden", "/Volumes"], sudo=True, check=False)
    _LOG.info("revealed ~/Library and /Volumes")


def revert_system(archive: Archive, runner: Runner, *, dry_run: bool = False) -> None:
    """Best-effort revert of system settings, guarded where the value reads back."""
    records = {record["name"]: record for record in archive.manifest.get("system", [])}
    home = Path.home()

    for item in _scalar_items(home):
        record = records.get(item.name)
        if record is None:
            continue
        current = item.parse(_read(runner, item.read, item.read_sudo))
        if current != record["applied"]:
            _LOG.warning("skip (user-modified) %s", item.name)
            continue
        for write in item.restore(record["original"], record["present"]):
            if not dry_run:
                runner.run(write, sudo=True)
        _LOG.info("revert %s", item.name)

    if "pmset" in records:
        _revert_pmset(records["pmset"], runner, dry_run=dry_run)
    if "chflags:library" in records and not dry_run:
        runner.run(["chflags", "hidden", str(home / "Library")], check=False)
        runner.run(["chflags", "hidden", "/Volumes"], sudo=True, check=False)
        _LOG.info("revert chflags (re-hidden, best-effort)")
    if "systemsetup:restartfreeze" in records:
        _LOG.warning("skip systemsetup:restartfreeze (no getter; left as-is)")


def _revert_pmset(record: dict[str, Any], runner: Runner, *, dry_run: bool) -> None:
    """Restore readable pmset keys from the snapshot; reset the rest to documented defaults."""
    before = record["original"]
    applied = record["applied"]
    current = parse_pmset_custom(_read(runner, ["pmset", "-g", "custom"], False))
    for section, flag in (("battery", "-b"), ("ac", "-c")):
        for key in _PMSET_READABLE:
            if key not in applied.get(section, {}):
                continue
            if current.get(section, {}).get(key) != applied[section][key]:
                _LOG.warning("skip (user-modified) pmset %s %s", flag, key)
                continue
            original = before.get(section, {}).get(key)
            if original is None:
                continue
            if not dry_run:
                runner.run(["pmset", flag, key, original], sudo=True)
            _LOG.info("revert pmset %s %s", flag, key)
    for key, default in _PMSET_DEFAULTS.items():
        if not dry_run:
            runner.run(["pmset", "-a", key, default], sudo=True)
        _LOG.warning("reset pmset %s to documented default %s (not precisely restorable)", key, default)
