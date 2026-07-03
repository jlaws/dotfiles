"""Tests for macos_setup.dotfiles."""

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from macos_setup.archive import Archive
from macos_setup.dotfiles import (
    apply_file,
    file_revert_decision,
    remove_file,
    revert_files,
    sha256_file,
)


def _noop(*_args):
    pass


class Sha256Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_matches_hashlib(self):
        path = self.tmp / "f"
        path.write_bytes(b"hello world")
        self.assertEqual(sha256_file(path), hashlib.sha256(b"hello world").hexdigest())


class RevertDecisionTests(unittest.TestCase):
    def test_replaced_match_restores(self):
        self.assertEqual(file_revert_decision("replaced", "aaa", "aaa"), "restore")

    def test_replaced_mismatch_skips(self):
        self.assertEqual(file_revert_decision("replaced", "aaa", "bbb"), "skip")

    def test_added_match_deletes(self):
        self.assertEqual(file_revert_decision("added", "aaa", "aaa"), "delete")

    def test_added_mismatch_skips(self):
        self.assertEqual(file_revert_decision("added", "aaa", "bbb"), "skip")

    def test_removed_absent_restores(self):
        self.assertEqual(file_revert_decision("removed", "aaa", None), "restore")

    def test_removed_present_skips(self):
        self.assertEqual(file_revert_decision("removed", "aaa", "aaa"), "skip")


class ApplyFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.archive = Archive.create(self.tmp / "arch", "ts")
        self.src = self.tmp / "src"
        self.src.write_text("new content")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_replaced_archives_original(self):
        dest = self.tmp / "home" / ".extra"
        dest.parent.mkdir(parents=True)
        dest.write_text("old content")

        apply_file(self.src, dest, self.archive)

        self.assertEqual(dest.read_text(), "new content")
        archived = self.archive.files_dir / str(dest).lstrip("/")
        self.assertEqual(archived.read_text(), "old content")
        rec = self.archive.manifest["files"][0]
        self.assertEqual(rec["action"], "replaced")
        self.assertEqual(rec["sha256"], sha256_file(self.src))

    def test_added_records_added(self):
        dest = self.tmp / "home" / ".newfile"

        apply_file(self.src, dest, self.archive)

        self.assertEqual(dest.read_text(), "new content")
        self.assertEqual(self.archive.manifest["files"][0]["action"], "added")


class RemoveFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.archive = Archive.create(self.tmp / "arch", "ts")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_existing_file_archived_and_removed(self):
        dest = self.tmp / "hooks" / "lessons-learned.sh"
        dest.parent.mkdir(parents=True)
        dest.write_text("stale")

        remove_file(dest, self.archive)

        self.assertFalse(dest.exists())
        self.assertEqual(self.archive.manifest["files"][0]["action"], "removed")

    def test_absent_file_records_nothing(self):
        remove_file(self.tmp / "nope.sh", self.archive)
        self.assertEqual(self.archive.manifest["files"], [])


class RevertFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.archive = Archive.create(self.tmp / "arch", "ts")
        self.src = self.tmp / "src"
        self.src.write_text("managed")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_replaced_restored(self):
        dest = self.tmp / "home" / ".extra"
        dest.parent.mkdir(parents=True)
        dest.write_text("original")
        apply_file(self.src, dest, self.archive)

        summary = revert_files(self.archive, log=_noop)

        self.assertEqual(dest.read_text(), "original")
        self.assertIn(str(dest), summary.restored)

    def test_added_deleted(self):
        dest = self.tmp / "home" / ".added"
        apply_file(self.src, dest, self.archive)

        summary = revert_files(self.archive, log=_noop)

        self.assertFalse(dest.exists())
        self.assertIn(str(dest), summary.deleted)

    def test_user_modified_skipped(self):
        dest = self.tmp / "home" / ".extra"
        dest.parent.mkdir(parents=True)
        dest.write_text("original")
        apply_file(self.src, dest, self.archive)
        dest.write_text("user edit after setup")

        summary = revert_files(self.archive, log=_noop)

        self.assertEqual(dest.read_text(), "user edit after setup")
        self.assertIn(str(dest), summary.skipped)


if __name__ == "__main__":
    unittest.main()
