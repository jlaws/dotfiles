"""Tests for macos_setup.archive."""

import shutil
import tempfile
import unittest
from pathlib import Path

from macos_setup.archive import Archive, resolve_archive


class ResolveArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_latest_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_archive(self.tmp, None)

    def test_explicit_ref_returns_dir(self):
        Archive.create(self.tmp, "2026-01-02-030405")
        got = resolve_archive(self.tmp, "2026-01-02-030405")
        self.assertEqual(got, self.tmp / "2026-01-02-030405")

    def test_none_returns_latest(self):
        Archive.create(self.tmp, "2026-01-02-030405")
        got = resolve_archive(self.tmp, None)
        self.assertEqual(got.name, "2026-01-02-030405")

    def test_unknown_ref_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_archive(self.tmp, "does-not-exist")


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_makes_dirs_and_latest_symlink(self):
        archive = Archive.create(self.tmp, "ts1")
        self.assertTrue((self.tmp / "ts1" / "files").is_dir())
        self.assertTrue((self.tmp / "ts1" / "domains").is_dir())
        self.assertTrue((self.tmp / "latest").exists())
        self.assertEqual((self.tmp / "latest").resolve().name, "ts1")
        self.assertEqual(archive.path, self.tmp / "ts1")

    def test_latest_symlink_moves_to_newest(self):
        Archive.create(self.tmp, "ts1")
        Archive.create(self.tmp, "ts2")
        self.assertEqual((self.tmp / "latest").resolve().name, "ts2")

    def test_manifest_roundtrip(self):
        archive = Archive.create(self.tmp, "ts1")
        archive.record_file("/home/user/.extra", "replaced", "abc123")
        archive.record_setting("user", "com.apple.dock", "tilesize", present=True, applied=36)
        archive.record_system(
            "nvram:SystemAudioVolume", present=True, original="%80", applied=" "
        )
        archive.save()

        loaded = Archive.load(self.tmp / "ts1")
        self.assertEqual(loaded.manifest["version"], 1)
        self.assertEqual(loaded.manifest["files"][0]["sha256"], "abc123")
        self.assertEqual(loaded.manifest["files"][0]["action"], "replaced")
        self.assertEqual(loaded.manifest["settings"][0]["key"], "tilesize")
        self.assertEqual(loaded.manifest["settings"][0]["applied"], 36)
        self.assertEqual(loaded.manifest["system"][0]["name"], "nvram:SystemAudioVolume")
        self.assertEqual(loaded.manifest["system"][0]["original"], "%80")


if __name__ == "__main__":
    unittest.main()
