"""Tests for macos_setup.system."""

import shutil
import tempfile
import unittest
from pathlib import Path

from macos_setup.archive import Archive
from macos_setup.shell import CompletedResult
from macos_setup.system import (
    apply_system,
    parse_firewall_state,
    parse_nvram_value,
    parse_pmset_custom,
    parse_timezone,
    revert_system,
)
from tests.fakes import FakeRunner

PMSET_SAMPLE = """Battery Power:
 displaysleep         2
 sleep                1
 disksleep            10
 powernap             1
 hibernatemode        3
AC Power:
 displaysleep         10
 sleep                1
 disksleep            10
 powernap             1
 hibernatemode        3
"""


class ParserTests(unittest.TestCase):
    def test_pmset_splits_battery_and_ac(self):
        parsed = parse_pmset_custom(PMSET_SAMPLE)
        self.assertEqual(parsed["battery"]["displaysleep"], "2")
        self.assertEqual(parsed["ac"]["displaysleep"], "10")
        self.assertEqual(parsed["battery"]["hibernatemode"], "3")

    def test_nvram_value(self):
        self.assertEqual(parse_nvram_value("SystemAudioVolume\t%80\n"), "%80")

    def test_nvram_absent(self):
        self.assertIsNone(parse_nvram_value(""))

    def test_firewall_state(self):
        self.assertEqual(parse_firewall_state("Firewall is disabled. (State = 0)"), 0)
        self.assertEqual(parse_firewall_state("Firewall is enabled. (State = 1)"), 1)

    def test_timezone(self):
        self.assertEqual(parse_timezone("Time Zone: America/New_York"), "America/New_York")


def _reader(argv):
    """Canned read responses for apply_system's capture calls."""
    if argv[:2] == ["pmset", "-g"]:
        return CompletedResult(0, PMSET_SAMPLE, "")
    if argv[:1] == ["nvram"]:
        return CompletedResult(0, "SystemAudioVolume\t%80\n", "")
    if "--getglobalstate" in argv:
        return CompletedResult(0, "Firewall is disabled. (State = 0)", "")
    if "--getstealthmode" in argv:
        return CompletedResult(0, "Stealth mode disabled", "")
    if "-gettimezone" in argv:
        return CompletedResult(0, "Time Zone: America/Chicago", "")
    if "-getcomputersleep" in argv:
        return CompletedResult(0, "Computer Sleep: 10", "")
    if "-getwakeonnetworkaccess" in argv:
        return CompletedResult(0, "Wake On Network Access: on", "")
    return None


class ApplySystemTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_records_originals_and_issues_writes(self):
        runner = FakeRunner(_reader)
        archive = Archive.create(self.tmp, "ts")

        apply_system(archive, runner, self.tmp)

        names = {r["name"]: r for r in archive.manifest["system"]}
        self.assertEqual(names["nvram:SystemAudioVolume"]["original"], "%80")
        self.assertEqual(names["firewall:globalstate"]["original"], 0)
        argvs = runner.argv_list()
        self.assertIn(["nvram", "SystemAudioVolume= "], argvs)
        self.assertIn(["pmset", "-a", "lidwake", "1"], argvs)

    def test_dry_run_issues_no_writes(self):
        runner = FakeRunner(_reader)
        archive = Archive.create(self.tmp, "ts")

        apply_system(archive, runner, self.tmp, dry_run=True)

        self.assertNotIn(["nvram", "SystemAudioVolume= "], runner.argv_list())


class RevertSystemTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_restores_nvram_when_guard_passes(self):
        archive = Archive.create(self.tmp, "ts")
        archive.record_system(
            "nvram:SystemAudioVolume", present=True, original="%80", applied=" "
        )
        archive.save()

        # current value equals applied (" "), so the guard passes.
        runner = FakeRunner(lambda argv: CompletedResult(0, "SystemAudioVolume\t \n", ""))
        revert_system(archive, runner)

        self.assertIn(["nvram", "SystemAudioVolume=%80"], runner.argv_list())

    def test_skips_nvram_when_user_modified(self):
        archive = Archive.create(self.tmp, "ts")
        archive.record_system(
            "nvram:SystemAudioVolume", present=True, original="%80", applied=" "
        )
        archive.save()

        # current value differs from applied, so leave it alone.
        runner = FakeRunner(lambda argv: CompletedResult(0, "SystemAudioVolume\t%50\n", ""))
        revert_system(archive, runner)

        self.assertNotIn(["nvram", "SystemAudioVolume=%80"], runner.argv_list())

    def test_user_modified_logs_warning(self):
        archive = Archive.create(self.tmp, "ts")
        archive.record_system(
            "nvram:SystemAudioVolume", present=True, original="%80", applied=" "
        )
        archive.save()
        runner = FakeRunner(lambda argv: CompletedResult(0, "SystemAudioVolume\t%50\n", ""))

        with self.assertLogs("macos_setup.system", level="WARNING") as cm:
            revert_system(archive, runner)
        self.assertTrue(any("user-modified" in line for line in cm.output))

    def test_restore_logs_info(self):
        archive = Archive.create(self.tmp, "ts")
        archive.record_system(
            "nvram:SystemAudioVolume", present=True, original="%80", applied=" "
        )
        archive.save()
        runner = FakeRunner(lambda argv: CompletedResult(0, "SystemAudioVolume\t \n", ""))

        with self.assertLogs("macos_setup.system", level="INFO") as cm:
            revert_system(archive, runner)
        self.assertTrue(any("revert" in line for line in cm.output))


if __name__ == "__main__":
    unittest.main()
