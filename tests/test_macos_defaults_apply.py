"""Integration tests for apply_defaults / revert_defaults using a stateful defaults fake."""

import plistlib
import shutil
import tempfile
import unittest
from pathlib import Path

from macos_setup.archive import Archive
from macos_setup.macos_defaults import Setting, apply_defaults, revert_defaults
from macos_setup.shell import CompletedResult
from tests.fakes import FakeRunner


def _parse_write(args):
    """Parse a scalar ``defaults write`` value list into a native value."""
    flag, val = args[0], args[1]
    if flag == "-bool":
        return val == "true"
    if flag == "-int":
        return int(val)
    if flag == "-float":
        return float(val)
    return val


class FakeDefaults:
    """Minimal in-memory model of the ``defaults`` command for export/write/import."""

    def __init__(self, initial):
        self.state = {d: dict(v) for d, v in initial.items()}

    def handle(self, argv):
        rest = argv[1:] if argv and argv[0] == "defaults" else None
        if rest is None:
            return None
        if rest and rest[0] == "-currentHost":
            rest = rest[1:]
        cmd = rest[0]
        if cmd == "export":
            domain = rest[1]
            data = self.state.get(domain)
            if data is None:
                return CompletedResult(1, "", "does not exist")
            return CompletedResult(0, plistlib.dumps(data).decode(), "")
        if cmd == "write":
            domain, key, args = rest[1], rest[2], rest[3:]
            self.state.setdefault(domain, {})[key] = _parse_write(args)
            return CompletedResult(0, "", "")
        if cmd == "import":
            # `defaults import` merges into the domain (verified against real `defaults`).
            domain, path = rest[1], rest[2]
            self.state.setdefault(domain, {}).update(plistlib.loads(Path(path).read_bytes()))
            return CompletedResult(0, "", "")
        if cmd == "delete":
            domain, key = rest[1], rest[2]
            self.state.get(domain, {}).pop(key, None)
            return CompletedResult(0, "", "")
        return None


class ApplyDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_apply_snapshots_and_records(self):
        fake = FakeDefaults({"com.apple.dock": {"tilesize": 48}})
        runner = FakeRunner(fake.handle)
        archive = Archive.create(self.tmp, "ts")
        settings = [
            Setting("com.apple.dock", "tilesize", "int", 36),
            Setting("com.apple.dock", "autohide", "bool", True),
        ]

        apply_defaults(settings, archive, runner, "14.0")

        snap = plistlib.loads((archive.domains_dir / "com.apple.dock.plist").read_bytes())
        self.assertEqual(snap, {"tilesize": 48})
        recs = {r["key"]: r for r in archive.manifest["settings"]}
        self.assertTrue(recs["tilesize"]["present"])
        self.assertEqual(recs["tilesize"]["applied"], 36)
        self.assertFalse(recs["autohide"]["present"])
        self.assertTrue(recs["autohide"]["applied"])
        self.assertIn(
            ["defaults", "write", "com.apple.dock", "tilesize", "-int", "36"],
            runner.argv_list(),
        )

    def test_dry_run_writes_nothing(self):
        fake = FakeDefaults({"com.apple.dock": {"tilesize": 48}})
        runner = FakeRunner(fake.handle)
        archive = Archive.create(self.tmp, "ts")

        apply_defaults(
            [Setting("com.apple.dock", "tilesize", "int", 36)],
            archive,
            runner,
            "14.0",
            dry_run=True,
        )

        self.assertEqual(fake.state["com.apple.dock"]["tilesize"], 48)
        self.assertFalse(any(c[:2] == ["defaults", "write"] for c in runner.argv_list()))


class RevertDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _apply(self, fake):
        runner = FakeRunner(fake.handle)
        archive = Archive.create(self.tmp, "ts")
        settings = [
            Setting("com.apple.dock", "tilesize", "int", 36),
            Setting("com.apple.dock", "autohide", "bool", True),
        ]
        apply_defaults(settings, archive, runner, "14.0")
        return archive, runner

    def test_revert_restores_original_and_deletes_added(self):
        fake = FakeDefaults({"com.apple.dock": {"tilesize": 48}})
        archive, runner = self._apply(fake)

        summary = revert_defaults(archive, runner)

        self.assertEqual(fake.state["com.apple.dock"]["tilesize"], 48)
        self.assertNotIn("autohide", fake.state["com.apple.dock"])
        self.assertIn(("com.apple.dock", "tilesize"), summary.reverted)
        self.assertIn(("com.apple.dock", "autohide"), summary.reverted)

    def test_revert_skips_user_modified(self):
        fake = FakeDefaults({"com.apple.dock": {"tilesize": 48}})
        archive, runner = self._apply(fake)
        fake.state["com.apple.dock"]["tilesize"] = 64  # user change after setup

        summary = revert_defaults(archive, runner)

        self.assertEqual(fake.state["com.apple.dock"]["tilesize"], 64)
        self.assertIn(("com.apple.dock", "tilesize"), summary.skipped)

    def test_skip_logs_warning(self):
        fake = FakeDefaults({"com.apple.dock": {"tilesize": 48}})
        archive, runner = self._apply(fake)
        fake.state["com.apple.dock"]["tilesize"] = 64

        with self.assertLogs("macos_setup.macos_defaults", level="WARNING") as cm:
            revert_defaults(archive, runner)
        self.assertTrue(any("user-modified" in line for line in cm.output))

    def test_apply_logs_info_per_domain(self):
        fake = FakeDefaults({})
        runner = FakeRunner(fake.handle)
        archive = Archive.create(Path(tempfile.mkdtemp()), "ts")

        with self.assertLogs("macos_setup.macos_defaults", level="INFO") as cm:
            apply_defaults(
                [Setting("com.apple.dock", "tilesize", "int", 36)], archive, runner, "14.0"
            )
        self.assertTrue(any("com.apple.dock" in line for line in cm.output))


if __name__ == "__main__":
    unittest.main()
