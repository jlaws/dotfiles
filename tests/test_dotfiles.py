"""Tests for macos_setup.dotfiles."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from macos_setup.archive import Archive
from macos_setup.config_merge import ConfigMergeError
from macos_setup.dotfiles import (
    AGENT_ORPHAN_GLOBS,
    AGENT_REMOVALS,
    AGENT_SYNC,
    _iter_pairs,
    apply_file,
    apply_merged_file,
    expand_home_anchors,
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


class ApplyMergedFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.archive = Archive.create(self.tmp / "arch", "ts")
        self.src = self.tmp / "repo_settings.json"
        self.src.write_text('{"outputStyle": "Concise", "newKey": true}\n')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_existing_target_is_archived_and_merged_with_dest_sha_recorded(self) -> None:
        dest = self.tmp / "home" / ".claude" / "settings.json"
        dest.parent.mkdir(parents=True)
        dest.write_text('{"outputStyle": "Verbose", "machineId": "mac1"}\n')

        apply_merged_file(self.src, dest, self.archive, format_type="json")

        content = json.loads(dest.read_text())
        self.assertEqual(content["outputStyle"], "Concise")
        self.assertEqual(content["machineId"], "mac1")
        self.assertTrue(content["newKey"])

        archived = self.archive.files_dir / str(dest).lstrip("/")
        self.assertEqual(archived.read_text(), '{"outputStyle": "Verbose", "machineId": "mac1"}\n')

        rec = self.archive.manifest["files"][0]
        self.assertEqual(rec["dest"], str(dest))
        self.assertEqual(rec["action"], "replaced")
        self.assertEqual(rec["sha256"], sha256_file(dest))
        self.assertNotEqual(rec["sha256"], sha256_file(self.src))

    def test_absent_target_is_created_and_recorded_as_added(self) -> None:
        dest = self.tmp / "home" / ".claude" / "settings.json"

        apply_merged_file(self.src, dest, self.archive, format_type="json")

        self.assertTrue(dest.exists())
        rec = self.archive.manifest["files"][0]
        self.assertEqual(rec["dest"], str(dest))
        self.assertEqual(rec["action"], "added")
        self.assertEqual(rec["sha256"], sha256_file(dest))

    def test_apply_merged_file_atomic_write(self) -> None:
        dest = self.tmp / "home" / ".claude" / "settings.json"
        dest.parent.mkdir(parents=True)
        dest.write_text('{"outputStyle": "Verbose"}\n')

        apply_merged_file(self.src, dest, self.archive, format_type="json")

        tmp_files = list(dest.parent.glob(".*.tmp"))
        self.assertEqual(tmp_files, [])
        self.assertTrue(dest.exists())

    def test_malformed_target_is_archived_and_overwritten_with_repo_template(self) -> None:
        dest = self.tmp / "home" / ".claude" / "settings.json"
        dest.parent.mkdir(parents=True)
        dest.write_text('{"invalid_json": ')

        apply_merged_file(self.src, dest, self.archive, format_type="json")

        self.assertEqual(dest.read_text(), self.src.read_text())
        archived = self.archive.files_dir / str(dest).lstrip("/")
        self.assertEqual(archived.read_text(), '{"invalid_json": ')

    def test_revert_restores_pre_merge_backup(self) -> None:
        dest = self.tmp / "home" / ".claude" / "settings.json"
        dest.parent.mkdir(parents=True)
        dest.write_text('{"outputStyle": "Verbose", "machineId": "mac1"}\n')

        apply_merged_file(self.src, dest, self.archive, format_type="json")
        summary = revert_files(self.archive)

        self.assertEqual(summary.restored, [str(dest)])
        self.assertEqual(
            dest.read_text(encoding="utf-8"), '{"outputStyle": "Verbose", "machineId": "mac1"}\n'
        )

    def test_existing_target_is_archived_and_merged_toml(self) -> None:
        src = self.tmp / "repo_config.toml"
        src.write_text('model = "gpt-6-astra"\n[agents]\nmax_threads = 4\n', encoding="utf-8")
        dest = self.tmp / "home" / ".codex" / "config.toml"
        dest.parent.mkdir(parents=True)
        dest.write_text(
            'model = "gpt-5"\nlocal_key = "val"\n[agents]\nmax_threads = 8\n', encoding="utf-8"
        )

        action = apply_merged_file(src, dest, self.archive, format_type="toml")

        self.assertEqual(action, "replaced")
        content = dest.read_text(encoding="utf-8")
        self.assertIn('model = "gpt-6-astra"', content)
        self.assertIn('local_key = "val"', content)
        self.assertIn("max_threads = 4", content)
        self.assertNotIn('model = "gpt-5"', content)

    def test_malformed_target_toml_is_archived_and_reset_to_template(self) -> None:
        src = self.tmp / "repo_config.toml"
        src.write_text('model = "gpt-6-astra"\n', encoding="utf-8")
        dest = self.tmp / "home" / ".codex" / "config.toml"
        dest.parent.mkdir(parents=True)
        dest.write_text("[unclosed header\nkey = 1\n", encoding="utf-8")

        action = apply_merged_file(src, dest, self.archive, format_type="toml")

        self.assertEqual(action, "reset")
        self.assertEqual(dest.read_text(encoding="utf-8"), src.read_text(encoding="utf-8"))
        archived = self.archive.files_dir / str(dest).lstrip("/")
        self.assertEqual(archived.read_text(encoding="utf-8"), "[unclosed header\nkey = 1\n")

    def test_corrupt_binary_target_is_archived_and_reset_to_template(self) -> None:
        dest = self.tmp / "home" / ".claude" / "settings.json"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"\x80\x81\x82 corrupt binary")

        action = apply_merged_file(self.src, dest, self.archive, format_type="json")

        self.assertEqual(action, "reset")
        self.assertEqual(dest.read_text(encoding="utf-8"), self.src.read_text(encoding="utf-8"))

    def test_unsupported_format_raises_value_error(self) -> None:
        dest = self.tmp / "home" / "config.yaml"
        with self.assertRaises(ValueError):
            apply_merged_file(self.src, dest, self.archive, format_type="yaml")

    def test_malformed_repo_source_raises_and_does_not_clobber_dest(self) -> None:
        bad_src = self.tmp / "bad_src.json"
        bad_src.write_text('{"unclosed": ', encoding="utf-8")
        dest = self.tmp / "home" / ".claude" / "settings.json"
        dest.parent.mkdir(parents=True)
        dest.write_text('{"keepMe": true}\n', encoding="utf-8")

        with self.assertRaises(ConfigMergeError):
            apply_merged_file(bad_src, dest, self.archive, format_type="json")

        self.assertEqual(dest.read_text(encoding="utf-8"), '{"keepMe": true}\n')


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

    def test_sync_logs_each_retired_permission(self) -> None:
        repo_settings = self.repo / ".claude" / "settings.json"
        repo_settings.parent.mkdir(parents=True, exist_ok=True)
        repo_settings.write_text('{"permissions": {"allow": ["Bash(git diff *)"]}}\n')
        target = self._write_target(
            ".claude/settings.json",
            '{"permissions": {"allow": ["Bash(npx *)", "Bash(gh *)", "Bash(ls *)"]}}\n',
        )

        with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
            sync_agents(self.repo, self.target, self.archive)

        output = "\n".join(captured.output)
        self.assertIn(f"retired Bash(npx *) from {target}", output)
        self.assertIn(f"retired Bash(gh *) from {target}", output)
        self.assertNotIn("retired Bash(ls *)", output)
        allow = json.loads(target.read_text())["permissions"]["allow"]
        self.assertEqual(allow, ["Bash(git diff *)", "Bash(ls *)"])

    def _repo_settings(self, relative: str, text: str) -> None:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_sync_retired_log_points_at_settings_local_and_counts_in_summary(self) -> None:
        self._repo_settings(".claude/settings.json", '{"permissions": {"allow": []}}\n')
        self._write_target(
            ".claude/settings.json",
            '{"permissions": {"allow": ["Bash(npx *)", "Bash(gh:*)"]}}\n',
        )
        with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
            sync_agents(self.repo, self.target, self.archive)
        self.assertTrue(
            any(
                "retired Bash(gh:*) from" in e and "settings.local.json" in e
                for e in captured.output
            ),
            captured.output,
        )
        self.assertTrue(any("2 retired" in e for e in captured.output), captured.output)

    def test_dry_run_previews_each_retirement_and_writes_nothing(self) -> None:
        self._repo_settings(".claude/settings.json", '{"permissions": {"allow": []}}\n')
        text = '{"permissions": {"allow": ["Bash(npx *)", "Bash(ls *)"]}}\n'
        target = self._write_target(".claude/settings.json", text)
        with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
            sync_agents(self.repo, self.target, self.archive, dry_run=True)
        self.assertTrue(
            any(f"would retire Bash(npx *) from {target}" in e for e in captured.output),
            captured.output,
        )
        self.assertFalse(any("would retire Bash(ls" in e for e in captured.output))
        self.assertTrue(any("1 would retire" in e for e in captured.output), captured.output)
        self.assertEqual(target.read_text(), text)

    def test_sync_retires_antigravity_command_entries(self) -> None:
        self._repo_settings(
            ".gemini/antigravity-cli/settings.json",
            '{"permissions": {"allow": ["command(git diff)"]}}\n',
        )
        target = self._write_target(
            ".gemini/antigravity-cli/settings.json",
            '{"permissions": {"allow": ["command(python3)", "command(gh)", "command(ls)"]}}\n',
        )
        with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
            sync_agents(self.repo, self.target, self.archive)
        output = "\n".join(captured.output)
        self.assertIn(f"retired command(python3) from {target}", output)
        self.assertIn(f"retired command(gh) from {target}", output)
        allow = json.loads(target.read_text())["permissions"]["allow"]
        self.assertEqual(allow, ["command(git diff)", "command(ls)"])

    def test_retired_is_not_logged_when_the_write_fails(self) -> None:
        from unittest import mock

        self._repo_settings(".claude/settings.json", '{"permissions": {"allow": []}}\n')
        target = self._write_target(
            ".claude/settings.json", '{"permissions": {"allow": ["Bash(npx *)"]}}\n'
        )
        with (
            mock.patch.object(Path, "replace", side_effect=OSError("boom")),
            self.assertLogs("macos_setup.dotfiles", level="INFO") as captured,
        ):
            import logging

            logging.getLogger("macos_setup.dotfiles").info("probe")
            with self.assertRaises(OSError):
                sync_agents(self.repo, self.target, self.archive)
        self.assertFalse(any("retired Bash(npx" in e for e in captured.output))
        self.assertIn("npx", target.read_text())

    def test_a_string_allow_in_the_target_logs_no_retirement(self) -> None:
        self._repo_settings(".claude/settings.json", '{"permissions": {"allow": []}}\n')
        self._write_target(".claude/settings.json", '{"permissions": {"allow": "Bash(npx *)"}}\n')
        with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
            sync_agents(self.repo, self.target, self.archive)
        self.assertFalse(any("retired Bash" in e for e in captured.output), captured.output)

    def test_sync_agents_merges_configs_and_replaces_skills(self) -> None:
        repo_skill = self.repo / ".agents" / "skills" / "cmd-j-tdd" / "SKILL.md"
        repo_skill.parent.mkdir(parents=True, exist_ok=True)
        repo_skill.write_text("---\nname: cmd-j-tdd\nversion: 2\n---\n")

        repo_claude_settings = self.repo / ".claude" / "settings.json"
        repo_claude_settings.parent.mkdir(parents=True, exist_ok=True)
        repo_claude_settings.write_text('{"outputStyle": "Concise", "newKey": true}\n')

        repo_codex_config = self.repo / ".codex" / "config.toml"
        repo_codex_config.parent.mkdir(parents=True, exist_ok=True)
        repo_codex_config.write_text('model = "gpt-6-astra"\n[agents]\nmax_threads = 6\n')

        target_skill = self._write_target(
            ".agents/skills/cmd-j-tdd/SKILL.md", "local custom skill edit"
        )
        target_claude_settings = self._write_target(
            ".claude/settings.json",
            '{"outputStyle": "Verbose", "enabledPlugins": {"my-plugin": true}}\n',
        )
        target_codex_config = self._write_target(
            ".codex/config.toml",
            'model = "gpt-5"\nmachine_id = "local1"\n[agents]\nmax_threads = 12\n',
        )

        sync_agents(self.repo, self.target, self.archive)

        self.assertIn("version: 2", target_skill.read_text())
        self.assertNotIn("local custom skill edit", target_skill.read_text())

        claude_res = json.loads(target_claude_settings.read_text())
        self.assertEqual(claude_res["outputStyle"], "Concise")
        self.assertTrue(claude_res["newKey"])
        self.assertTrue(claude_res["enabledPlugins"]["my-plugin"])

        codex_res = target_codex_config.read_text()
        self.assertIn('model = "gpt-6-astra"', codex_res)
        self.assertIn('machine_id = "local1"', codex_res)

    def test_sync_agents_dry_run_previews_merge_without_writing(self) -> None:
        target_claude = self._write_target(".claude/settings.json", '{"outputStyle": "Verbose"}\n')
        repo_claude = self.repo / ".claude" / "settings.json"
        repo_claude.parent.mkdir(parents=True, exist_ok=True)
        repo_claude.write_text('{"outputStyle": "Concise"}\n')

        with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
            sync_agents(self.repo, self.target, self.archive, dry_run=True)

        self.assertTrue(
            any(
                "would merge" in line and ".claude/settings.json" in line
                for line in captured.output
            )
        )
        self.assertEqual(target_claude.read_text(), '{"outputStyle": "Verbose"}\n')
        self.assertEqual(len(self.archive.manifest["files"]), 0)

    def test_sync_agents_revert_restores_pre_merge_backup(self) -> None:
        target_claude = self._write_target(".claude/settings.json", '{"local_setting": true}\n')
        repo_claude = self.repo / ".claude" / "settings.json"
        repo_claude.parent.mkdir(parents=True, exist_ok=True)
        repo_claude.write_text('{"repo_setting": true}\n')

        sync_agents(self.repo, self.target, self.archive)
        self.assertIn("repo_setting", target_claude.read_text())

        summary = revert_files(self.archive)
        self.assertIn(str(target_claude), summary.restored)
        self.assertEqual(target_claude.read_text(encoding="utf-8"), '{"local_setting": true}\n')

    def test_sync_agents_merges_gemini_settings(self) -> None:
        repo_gemini = self.repo / ".gemini" / "antigravity-cli" / "settings.json"
        repo_gemini.parent.mkdir(parents=True, exist_ok=True)
        repo_gemini.write_text('{"permissions": {"allow": ["command(git)"]}}\n', encoding="utf-8")

        target_gemini = self._write_target(
            ".gemini/antigravity-cli/settings.json",
            '{"permissions": {"allow": ["command(custom)"]}}\n',
        )

        sync_agents(self.repo, self.target, self.archive)

        data = json.loads(target_gemini.read_text(encoding="utf-8"))
        self.assertIn("command(git)", data["permissions"]["allow"])
        self.assertIn("command(custom)", data["permissions"]["allow"])

    def test_sync_agents_dry_run_previews_would_sync_when_target_absent(self) -> None:
        repo_claude = self.repo / ".claude" / "settings.json"
        repo_claude.parent.mkdir(parents=True, exist_ok=True)
        repo_claude.write_text('{"outputStyle": "Concise"}\n', encoding="utf-8")

        with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
            sync_agents(self.repo, self.target, self.archive, dry_run=True)

        self.assertTrue(
            any(
                "would sync" in line and ".claude/settings.json" in line for line in captured.output
            )
        )

    def test_sync_agents_revert_skips_user_modified_config(self) -> None:
        target_claude = self._write_target(".claude/settings.json", '{"local_setting": true}\n')
        repo_claude = self.repo / ".claude" / "settings.json"
        repo_claude.parent.mkdir(parents=True, exist_ok=True)
        repo_claude.write_text('{"repo_setting": true}\n', encoding="utf-8")

        sync_agents(self.repo, self.target, self.archive)
        target_claude.write_text('{"user_customized": true}\n', encoding="utf-8")

        summary = revert_files(self.archive)
        self.assertIn(str(target_claude), summary.skipped)
        self.assertEqual(target_claude.read_text(encoding="utf-8"), '{"user_customized": true}\n')

    def test_removes_cursor_agent_wrappers_and_orphaned_commands(self):
        cursor_file = self._write_target(".cursor/ai-tracking/tracking.db")
        agent_wrapper = self._write_target(".agents/skills/agent-reviewer/SKILL.md")
        current_command = self._write_target(".agents/skills/cmd-j-tdd/SKILL.md", "stale command")
        orphaned_command = self._write_target(".agents/skills/cmd-j-write-plan/SKILL.md")

        sync_agents(self.repo, self.target, self.archive)

        self.assertFalse(cursor_file.exists())
        self.assertFalse(agent_wrapper.exists())
        self.assertFalse(orphaned_command.exists())
        self.assertTrue(current_command.exists())
        self.assertIn("name: cmd-j-tdd", current_command.read_text())

    def _write_repo(self, relative: str, content: str = "shipped") -> Path:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def _orphan_entry(self, root: str, pattern: str, stem: str) -> str:
        """Return the relative path of a command entry in ``root`` for glob ``pattern``."""
        name = pattern.replace("*", stem)
        return f"{root}/{name}/SKILL.md" if not name.endswith(".md") else f"{root}/{name}"

    def test_removes_a_command_the_repo_no_longer_ships_from_each_tree(self):
        for root, pattern in AGENT_ORPHAN_GLOBS:
            with self.subTest(root=root):
                keep = self._orphan_entry(root, pattern, "keep")
                gone = self._orphan_entry(root, pattern, "gone")
                self._write_repo(keep)
                kept = self._write_target(keep)
                removed = self._write_target(gone)

                sync_agents(self.repo, self.target, self.archive)

                self.assertFalse(removed.exists())
                self.assertTrue(kept.exists())
                records = self.archive.manifest["files"]
                self.assertTrue(
                    any(r["dest"] == str(removed) and r["action"] == "removed" for r in records)
                )

    def test_removes_the_retired_category_named_codex_prompts(self):
        retired = self._write_target(".codex/prompts/workflow-diff-review.md")
        other = self._write_target(".codex/prompts/ai-ml-experiment.md")

        sync_agents(self.repo, self.target, self.archive)

        self.assertFalse(retired.exists())
        self.assertFalse(other.exists())

    def test_dry_run_reports_pruned_commands_without_removing_them(self):
        for root, pattern in AGENT_ORPHAN_GLOBS:
            with self.subTest(root=root):
                self._write_repo(self._orphan_entry(root, pattern, "keep"))
                gone = self._write_target(self._orphan_entry(root, pattern, "gone"))

                with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
                    sync_agents(self.repo, self.target, self.archive, dry_run=True)

                self.assertTrue(gone.exists())
                self.assertTrue(
                    any(
                        "would remove" in line and str(gone.parent) in line
                        for line in captured.output
                    ),
                    captured.output,
                )

    def test_keeps_a_user_authored_command_the_repo_never_shipped(self):
        notes = [
            self._write_target(".claude/commands/my-notes.md"),
            self._write_target(".codex/prompts/my-notes.md"),
            self._write_target(".gemini/antigravity-cli/skills/my-skill/SKILL.md"),
        ]

        sync_agents(self.repo, self.target, self.archive)

        for path in notes:
            self.assertTrue(path.exists(), path)

    def test_keeps_shared_gemini_skills(self):
        # A non-command skill that lives only in the target (a user's own, or a shared skill
        # synced earlier) must survive: only `j-*` entries are command skills.
        target = self._write_target(".gemini/antigravity-cli/skills/tdd/SKILL.md")

        sync_agents(self.repo, self.target, self.archive)

        self.assertTrue(target.exists())

    def test_keeps_the_codex_prompts_readme(self):
        readme = self._write_target(".codex/prompts/README.md")

        sync_agents(self.repo, self.target, self.archive)

        self.assertTrue(readme.exists())

    def test_orphan_prune_keeps_a_symlinked_entry(self):
        real = self.tmp / "elsewhere"
        real.mkdir()
        link = self.target / ".gemini/antigravity-cli/skills/j-linked"
        link.parent.mkdir(parents=True)
        link.symlink_to(real, target_is_directory=True)

        self._write_repo(".gemini/antigravity-cli/skills/j-keep/SKILL.md")

        with self.assertLogs("macos_setup.dotfiles", level="WARNING") as captured:
            sync_agents(self.repo, self.target, self.archive)

        self.assertTrue(link.is_symlink())
        self.assertTrue(real.is_dir())
        self.assertIn(
            f"keeping symlinked command {link}: not shipped by the repo; "
            "remove it by hand to stop this warning",
            " ".join(captured.output),
        )

    def test_dry_run_reports_removals_without_changing_files(self):
        cursor_file = self._write_target(".cursor/state.json")
        agent_wrapper = self._write_target(".agents/skills/agent-reviewer/SKILL.md")

        with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
            sync_agents(self.repo, self.target, self.archive, dry_run=True)

        self.assertTrue(cursor_file.exists())
        self.assertTrue(agent_wrapper.exists())
        self.assertTrue(any("would remove" in line for line in captured.output))
        self.assertEqual(self.archive.manifest["files"], [])

    def test_removes_legacy_gemini_artifacts_and_syncs_antigravity(self):
        legacy_hook = self._write_target(".gemini/hooks/log-prompt.sh", "#!/bin/sh")
        legacy_policy = self._write_target(".gemini/policies/default.toml", "deny = []")
        legacy_settings = self._write_target(".gemini/settings.json", "{}")
        legacy_command = self._write_target(".gemini/commands/j-plan.toml", "description = 'plan'")
        legacy_agent = self._write_target(".gemini/agents/code-reviewer.md", "# Agent")

        ag_settings = self.repo / ".gemini" / "antigravity-cli" / "settings.json"
        ag_settings.parent.mkdir(parents=True)
        ag_settings.write_text('{"permissions": {}}')

        ag_skill = self.repo / ".gemini" / "antigravity-cli" / "skills" / "j-plan" / "SKILL.md"
        ag_skill.parent.mkdir(parents=True)
        ag_skill.write_text("---\nname: j-plan\ndescription: plan\n---\n")

        ag_agent = self.repo / ".gemini" / "config" / "agents" / "code-reviewer.md"
        ag_agent.parent.mkdir(parents=True)
        ag_agent.write_text("---\nname: code-reviewer\nsubagent: true\nmainAgent: true\n---\n")

        sync_agents(self.repo, self.target, self.archive)

        self.assertFalse(legacy_hook.exists())
        self.assertFalse(legacy_policy.exists())
        self.assertFalse(legacy_settings.exists())
        self.assertFalse(legacy_command.exists())
        self.assertFalse(legacy_agent.exists())
        self.assertFalse((self.target / ".gemini" / "hooks").exists())
        self.assertFalse((self.target / ".gemini" / "policies").exists())
        self.assertFalse((self.target / ".gemini" / "commands").exists())
        self.assertFalse((self.target / ".gemini" / "agents").exists())

        installed_settings = self.target / ".gemini" / "antigravity-cli" / "settings.json"
        installed_skill = (
            self.target / ".gemini" / "antigravity-cli" / "skills" / "j-plan" / "SKILL.md"
        )
        installed_agent = self.target / ".gemini" / "config" / "agents" / "code-reviewer.md"
        self.assertTrue(installed_settings.exists())
        self.assertEqual(installed_settings.read_text(), '{"permissions": {}}')
        self.assertTrue(installed_skill.exists())
        self.assertIn("name: j-plan", installed_skill.read_text())
        self.assertTrue(installed_agent.exists())
        self.assertIn("subagent: true", installed_agent.read_text())

    def test_removes_the_retired_bash_output_guard_from_both_hook_trees(self):
        """The guard was deleted from the repo, but `.claude/hooks` and `.codex/hooks` sync as
        directories and syncing does not prune -- so without an AGENT_REMOVALS entry the installed
        copies would keep firing forever. See
        `docs/adr/context-efficiency/hooks-do-not-restate-loaded-guidance.md`.

        Archived before deletion, so `--uninstall` can put them back.
        """
        claude_guard = self._write_target(
            ".claude/hooks/guard-bash-output.sh", "#!/usr/bin/env bash"
        )
        codex_guard = self._write_target(".codex/hooks/guard-bash-output.sh", "#!/usr/bin/env bash")
        kept = self._write_target(".claude/hooks/log-prompt.sh", "#!/usr/bin/env bash")

        sync_agents(self.repo, self.target, self.archive)

        self.assertFalse(claude_guard.exists())
        self.assertFalse(codex_guard.exists())
        self.assertTrue(kept.exists())

        removed = {
            entry["dest"]
            for entry in self.archive.manifest["files"]
            if entry["action"] == "removed"
        }
        self.assertIn(str(claude_guard), removed)
        self.assertIn(str(codex_guard), removed)

    def test_a_live_removal_is_reported_at_info_like_the_preview(self):
        """`--dry-run` logged every path it would remove while the real run logged nothing at the
        default level -- `remove_file` reports at DEBUG. Two files left the user's home directory
        with no run output and only the archive manifest as evidence. The preview must not be
        louder than the run it previews.
        """
        guard = self._write_target(".claude/hooks/guard-bash-output.sh", "#!/usr/bin/env bash")

        with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
            sync_agents(self.repo, self.target, self.archive)

        self.assertFalse(guard.exists())
        self.assertTrue(
            any(f"removed stale {guard}" in line for line in captured.output),
            captured.output,
        )

    def test_removal_logs_carry_a_reason(self):
        self._write_repo(".claude/commands/j-keep.md")
        retired = self._write_target(".claude/hooks/guard-bash-output.sh")
        glob_hit = self._write_target(".agents/skills/agent-x/SKILL.md").parent
        orphan = self._write_target(".claude/commands/j-gone.md")

        with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
            sync_agents(self.repo, self.target, self.archive)

        def line(path):
            return next(entry for entry in captured.output if f"removed stale {path}" in entry)

        self.assertIn("retired", line(retired))
        self.assertIn("retired glob", line(glob_hit))
        self.assertIn("not shipped", line(orphan))

    def test_sync_summary_counts_removals(self):
        self._write_repo(".claude/commands/j-keep.md")
        self._write_target(".claude/commands/j-gone.md")
        self._write_target(".claude/hooks/guard-bash-output.sh")

        with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
            sync_agents(self.repo, self.target, self.archive)
        self.assertTrue(any("2 removed" in e for e in captured.output), captured.output)

        self._write_target(".claude/commands/j-gone.md")
        with self.assertLogs("macos_setup.dotfiles", level="INFO") as captured:
            sync_agents(self.repo, self.target, self.archive, dry_run=True)
        self.assertTrue(
            any("Would sync" in e and "1 would remove" in e for e in captured.output),
            captured.output,
        )

    def test_removal_skips_paths_behind_a_symlinked_parent(self):
        outside = self.tmp / "outside"
        outside.mkdir()
        victim = outside / "j-victim.md"
        victim.write_text("not ours")
        hook = outside / "guard-bash-output.sh"
        hook.write_text("not ours")
        self._write_repo(".claude/commands/j-keep.md")
        (self.target / ".claude").mkdir(parents=True)
        (self.target / ".claude/commands").symlink_to(outside, target_is_directory=True)
        (self.target / ".claude/hooks").symlink_to(outside, target_is_directory=True)

        with self.assertLogs("macos_setup.dotfiles", level="WARNING") as captured:
            sync_agents(self.repo, self.target, self.archive)

        self.assertTrue(victim.exists())
        self.assertTrue(hook.exists())
        self.assertTrue(any("symlink" in e and "commands" in e for e in captured.output))

    def test_a_root_the_repo_ships_nothing_in_is_not_pruned(self):
        stale = self._write_target(".claude/commands/j-user.md")

        with self.assertLogs("macos_setup.dotfiles", level="WARNING") as captured:
            sync_agents(self.repo, self.target, self.archive)

        self.assertTrue(stale.exists())
        self.assertTrue(
            any(
                "not pruning .claude/commands: the repo ships no j-*.md there" in e
                for e in captured.output
            ),
            captured.output,
        )

    def test_a_failed_removal_warns_and_the_sync_continues(self):
        self._write_repo(".gemini/antigravity-cli/skills/j-keep/SKILL.md")
        broken = self._write_target(".gemini/antigravity-cli/skills/j-bad/SKILL.md").parent
        (broken / "dangling").symlink_to(self.tmp / "missing")
        later = self._write_target(".gemini/antigravity-cli/skills/j-zzz/SKILL.md")
        hook = self._write_target(".claude/hooks/guard-bash-output.sh")

        with self.assertLogs("macos_setup.dotfiles", level="WARNING") as captured:
            sync_agents(self.repo, self.target, self.archive)

        self.assertTrue(any("j-bad" in e and "could not remove" in e for e in captured.output))
        self.assertFalse(later.exists())
        self.assertFalse(hook.exists())

    def test_orphan_globs_cover_the_four_command_trees(self):
        self.assertEqual(
            AGENT_ORPHAN_GLOBS,
            [
                (".agents/skills", "cmd-j-*"),
                (".claude/commands", "j-*.md"),
                (".codex/prompts", "j-*.md"),
                (".gemini/antigravity-cli/skills", "j-*"),
            ],
        )

    def test_no_retired_path_names_a_file_the_repo_ships(self):
        repo_root = Path(__file__).resolve().parent.parent
        shipped = {
            dest.relative_to(Path("/t"))
            for src_rel, dest_rel in AGENT_SYNC
            for _, dest in _iter_pairs(repo_root / src_rel, Path("/t") / dest_rel)
        }
        for removal in AGENT_REMOVALS:
            with self.subTest(removal=removal):
                path = Path(removal)
                self.assertFalse(
                    any(item == path or path in item.parents for item in shipped),
                    f"{removal} is retired but the repo ships it",
                )

    def test_a_user_authored_j_command_the_repo_does_not_ship_is_removed(self):
        # `j-` is reserved for repo-shipped commands; the prune cannot tell a user's j-mine.md
        # from a deleted one, so it removes both (archived, restorable).
        self._write_repo(".claude/commands/j-keep.md")
        mine = self._write_target(".claude/commands/j-mine.md", "my own command")

        sync_agents(self.repo, self.target, self.archive)

        self.assertFalse(mine.exists())

    def test_the_orphan_glob_removes_the_retired_finalize_pr_command(self):
        self._write_repo(".claude/commands/j-keep.md")
        stale = self._write_target(".claude/commands/j-finalize-pr.md")

        sync_agents(self.repo, self.target, self.archive)

        self.assertFalse(stale.exists())


class ExpandHomeAnchorsTests(unittest.TestCase):
    def test_expands_both_reference_prefixes(self):
        text = "Read `~/.agents/references/a/b.md` and `~/.claude/references/c/d.md`."
        got = expand_home_anchors(text, Path("/Users/x"))
        self.assertEqual(
            got,
            "Read `/Users/x/.agents/references/a/b.md` and `/Users/x/.claude/references/c/d.md`.",
        )

    def test_leaves_other_text_alone(self):
        text = "Skills live in ~/.agents/skills/ and ${CLAUDE_SKILL_DIR}/../../references/a/"
        self.assertEqual(expand_home_anchors(text, Path("/Users/x")), text)


class ExpandedAnchorSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "repo"
        self.home = self.tmp / "home"
        self.archive = Archive.create(self.tmp / "arch", "ts")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, relative: str, text: str) -> None:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_installed_assets_cite_references_by_absolute_home_path(self):
        self._write(".agents/skills/a/SKILL.md", "See `~/.agents/references/x/y.md`.\n")
        self._write(".codex/agents/a.toml", 'developer_instructions = "~/.agents/references/x/"\n')
        self._write(".claude/commands/c.md", "See `~/.claude/references/x/y.md`.\n")
        self._write(".claude/skills/s/SKILL.md", "See ${CLAUDE_SKILL_DIR}/../../references/x/\n")

        sync_agents(self.repo, self.home, self.archive)

        skill = (self.home / ".agents/skills/a/SKILL.md").read_text()
        self.assertEqual(skill, f"See `{self.home}/.agents/references/x/y.md`.\n")
        toml = (self.home / ".codex/agents/a.toml").read_text()
        self.assertIn(f'"{self.home}/.agents/references/x/"', toml)
        command = (self.home / ".claude/commands/c.md").read_text()
        self.assertEqual(command, f"See `{self.home}/.claude/references/x/y.md`.\n")
        sibling = (self.home / ".claude/skills/s/SKILL.md").read_text()
        self.assertIn("${CLAUDE_SKILL_DIR}/../../references/x/", sibling)
        for installed in self.home.rglob("*"):
            if installed.is_file():
                self.assertNotIn("~/.agents/references/", installed.read_text())
                self.assertNotIn("~/.claude/references/", installed.read_text())

    def test_uninstall_restores_a_file_whose_anchors_were_expanded(self):
        self._write(".agents/skills/a/SKILL.md", "See `~/.agents/references/x/y.md`.\n")
        original = self.home / ".agents/skills/a/SKILL.md"
        original.parent.mkdir(parents=True)
        original.write_text("user original\n")

        sync_agents(self.repo, self.home, self.archive)
        self.assertNotEqual(original.read_text(), "user original\n")
        summary = revert_files(self.archive)

        self.assertEqual(original.read_text(), "user original\n")
        self.assertIn(str(original), summary.restored)

    def test_uninstall_deletes_an_added_file_whose_anchors_were_expanded(self):
        self._write(".claude/agents/a.md", "See `~/.claude/references/x/y.md`.\n")
        sync_agents(self.repo, self.home, self.archive)
        added = self.home / ".claude/agents/a.md"

        summary = revert_files(self.archive)

        self.assertFalse(added.exists())
        self.assertIn(str(added), summary.deleted)

    def test_non_text_assets_are_copied_byte_for_byte(self):
        self._write(".agents/skills/a/agents/openai.yaml", "path: ~/.agents/references/x/\n")
        sync_agents(self.repo, self.home, self.archive)
        copied = (self.home / ".agents/skills/a/agents/openai.yaml").read_text()
        self.assertEqual(copied, "path: ~/.agents/references/x/\n")


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
