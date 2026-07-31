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
    remove_path,
    revert_files,
    sha256_file,
    sync_agents,
    sync_dotfiles,
)


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


class RemovePathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.archive = Archive.create(self.tmp / "arch", "ts")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_directory_files_are_archived_and_removed(self):
        dest = self.tmp / "home" / ".cursor"
        (dest / "ai-tracking").mkdir(parents=True)
        (dest / "settings.json").write_text("settings")
        (dest / "ai-tracking" / "tracking.db").write_text("database")

        remove_path(dest, self.archive)

        self.assertFalse(dest.exists())
        records = self.archive.manifest["files"]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["action"] == "removed" for record in records))

    def test_removed_directory_files_can_be_restored(self):
        dest = self.tmp / "home" / ".cursor"
        (dest / "nested").mkdir(parents=True)
        original = dest / "nested" / "state.json"
        original.write_text("state")
        remove_path(dest, self.archive)

        summary = revert_files(self.archive)

        self.assertEqual(original.read_text(), "state")
        self.assertIn(str(original), summary.restored)


class SyncDotfilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "repo"
        self.home = self.tmp / "home"
        self.archive = Archive.create(self.tmp / "arch", "ts")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_syncs_vim_runtime_files(self):
        colorscheme = self.repo / ".vim" / "colors" / "solarized.vim"
        colorscheme.parent.mkdir(parents=True)
        colorscheme.write_text("solarized")

        sync_dotfiles(self.repo, self.home, self.archive)

        installed = self.home / ".vim" / "colors" / "solarized.vim"
        self.assertEqual(installed.read_text(), "solarized")

    def test_creates_vim_state_directories(self):
        for name in ("backup", "undo", "swap"):
            marker = self.repo / ".vim" / name / ".gitkeep"
            marker.parent.mkdir(parents=True)
            marker.touch()

        sync_dotfiles(self.repo, self.home, self.archive)

        for name in ("backup", "undo", "swap"):
            self.assertTrue((self.home / ".vim" / name).is_dir())


class SyncAgentsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "repo"
        self.target = self.tmp / "home"
        self.archive = Archive.create(self.tmp / "arch", "ts")
        command = self.repo / ".agents" / "skills" / "cmd-j-tdd" / "SKILL.md"
        command.parent.mkdir(parents=True)
        command.write_text("---\nname: cmd-j-tdd\ndescription: Use when invoking TDD\n---\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_target(self, relative: str, content: str = "managed") -> Path:
        path = self.target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_removes_cursor_agent_wrappers_and_orphaned_commands(self):
        cursor_file = self._write_target(".cursor/ai-tracking/tracking.db")
        agent_wrapper = self._write_target(
            ".agents/skills/agent-reviewer/SKILL.md"
        )
        current_command = self._write_target(
            ".agents/skills/cmd-j-tdd/SKILL.md", "stale command"
        )
        orphaned_command = self._write_target(
            ".agents/skills/cmd-j-write-plan/SKILL.md"
        )

        sync_agents(self.repo, self.target, self.archive)

        self.assertFalse(cursor_file.exists())
        self.assertFalse(agent_wrapper.exists())
        self.assertFalse(orphaned_command.exists())
        self.assertTrue(current_command.exists())
        self.assertIn("name: cmd-j-tdd", current_command.read_text())

    def test_dry_run_reports_removals_without_changing_files(self):
        cursor_file = self._write_target(".cursor/state.json")
        agent_wrapper = self._write_target(
            ".agents/skills/agent-reviewer/SKILL.md"
        )

        with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
            sync_agents(self.repo, self.target, self.archive, dry_run=True)

        self.assertTrue(cursor_file.exists())
        self.assertTrue(agent_wrapper.exists())
        self.assertTrue(any("would remove" in line for line in captured.output))
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

        summary = revert_files(self.archive)

        self.assertEqual(dest.read_text(), "original")
        self.assertIn(str(dest), summary.restored)

    def test_added_deleted(self):
        dest = self.tmp / "home" / ".added"
        apply_file(self.src, dest, self.archive)

        summary = revert_files(self.archive)

        self.assertFalse(dest.exists())
        self.assertIn(str(dest), summary.deleted)

    def test_user_modified_skipped(self):
        dest = self.tmp / "home" / ".extra"
        dest.parent.mkdir(parents=True)
        dest.write_text("original")
        apply_file(self.src, dest, self.archive)
        dest.write_text("user edit after setup")

        summary = revert_files(self.archive)

        self.assertEqual(dest.read_text(), "user edit after setup")
        self.assertIn(str(dest), summary.skipped)

    def test_user_modified_logs_warning(self):
        dest = self.tmp / "home" / ".extra"
        dest.parent.mkdir(parents=True)
        dest.write_text("original")
        apply_file(self.src, dest, self.archive)
        dest.write_text("user edit after setup")

        with self.assertLogs("macos_setup.dotfiles", level="WARNING") as cm:
            revert_files(self.archive)
        self.assertTrue(any("user-modified" in line for line in cm.output))

    def test_restore_logs_info(self):
        dest = self.tmp / "home" / ".extra"
        dest.parent.mkdir(parents=True)
        dest.write_text("original")
        apply_file(self.src, dest, self.archive)

        with self.assertLogs("macos_setup.dotfiles", level="INFO") as cm:
            revert_files(self.archive)
        self.assertTrue(any("restore" in line for line in cm.output))


if __name__ == "__main__":
    unittest.main()
