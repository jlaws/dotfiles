from __future__ import annotations

import json
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

try:
    tomllib = import_module("tomllib")
except ImportError:  # Python 3.9 and 3.10 lack the optional independent parser.
    tomllib = None

from macos_setup.archive import Archive
from macos_setup.config_merge import ConfigMergeError, merge_json, merge_toml
from macos_setup.dotfiles import apply_merged_file


class MergeTomlTests(unittest.TestCase):
    def test_updates_existing_scalar_keys(self) -> None:
        target = 'model = "gpt-5"\nmodel_reasoning_effort = "low"\n'
        repo = 'model = "gpt-6-astra"\nmodel_reasoning_effort = "high"\n'
        res = merge_toml(repo, target)
        self.assertIn('model = "gpt-6-astra"', res)
        self.assertIn('model_reasoning_effort = "high"', res)
        self.assertNotIn('model = "gpt-5"', res)
        self.assertNotIn('model_reasoning_effort = "low"', res)

    def test_preserves_target_only_scalar_keys(self) -> None:
        target = 'machine_name = "work-laptop"\nmodel = "gpt-5"\n'
        repo = 'model = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn('machine_name = "work-laptop"', res)
        self.assertIn('model = "gpt-6-astra"', res)
        self.assertNotIn('model = "gpt-5"', res)

    def test_merges_tables_and_preserves_target_table_keys(self) -> None:
        target = "[agents]\nmax_threads = 8\nlocal_agent_flag = true\n"
        repo = '[agents]\ndefault_subagent_model = "gpt-6-sol"\nmax_threads = 6\n'
        res = merge_toml(repo, target)
        self.assertIn("[agents]", res)
        self.assertIn('default_subagent_model = "gpt-6-sol"', res)
        self.assertIn("max_threads = 6", res)
        self.assertIn("local_agent_flag = true", res)
        self.assertNotIn("max_threads = 8", res)

    def test_preserves_target_only_sections(self) -> None:
        target = '[mcp_servers.local]\ncommand = "my-mcp"\n'
        repo = 'model = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn("[mcp_servers.local]", res)
        self.assertIn('command = "my-mcp"', res)
        self.assertIn('model = "gpt-6-astra"', res)

    def test_appends_repo_only_sections(self) -> None:
        target = 'model = "gpt-5"\n'
        repo = 'model = "gpt-6-astra"\n[agents]\nmax_threads = 4\n'
        res = merge_toml(repo, target)
        self.assertIn("[agents]", res)
        self.assertIn("max_threads = 4", res)

    def test_preserves_comments_and_blank_lines(self) -> None:
        target = '# Machine specific comment\nmodel = "gpt-5"\n'
        repo = '# Repo hook comment\nmodel = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn("# Machine specific comment", res)
        self.assertIn('model = "gpt-6-astra"', res)

    def test_handles_section_headers_with_inline_comments(self) -> None:
        target = "[agents] # host settings\nmax_threads = 8\n"
        repo = '[agents]\ndefault_subagent_model = "gpt-6-sol"\nmax_threads = 4\n'
        res = merge_toml(repo, target)
        self.assertIn("[agents]", res)
        self.assertIn("max_threads = 4", res)
        self.assertIn('default_subagent_model = "gpt-6-sol"', res)
        # Should not duplicate table header
        self.assertEqual(res.count("[agents]"), 1)

    def test_multiline_array_with_quotes_and_comments(self) -> None:
        target = (
            'patterns = [\n  "[a-z]", # comment with ] bracket\n  "normal",\n]\nmodel = "gpt-5"\n'
        )
        repo = 'model = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn('"[a-z]"', res)
        self.assertIn('model = "gpt-6-astra"', res)

    def test_multiline_triple_quoted_strings(self) -> None:
        target = 'prompt = """\nhello [\nworld\n"""\nmodel = "gpt-5"\n'
        repo = 'model = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn("hello [", res)
        self.assertIn('model = "gpt-6-astra"', res)

    def test_array_of_tables_hooks_replaced_without_duplication(self) -> None:
        target = (
            "[[hooks.UserPromptSubmit]]\n\n"
            "[[hooks.UserPromptSubmit.hooks]]\n"
            'type = "command"\n'
            'command = "bash ~/.codex/hooks/old.sh"\n'
        )
        repo = (
            "[[hooks.UserPromptSubmit]]\n\n"
            "[[hooks.UserPromptSubmit.hooks]]\n"
            'type = "command"\n'
            'command = "bash ~/.codex/hooks/log-prompt.sh"\n'
        )
        res = merge_toml(repo, target)
        self.assertIn("bash ~/.codex/hooks/log-prompt.sh", res)
        self.assertNotIn("bash ~/.codex/hooks/old.sh", res)

    def test_preserves_target_only_array_of_tables(self) -> None:
        target = '[[custom_plugin.servers]]\nname = "local"\n'
        repo = 'model = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn("[[custom_plugin.servers]]", res)
        self.assertIn('name = "local"', res)
        self.assertIn('model = "gpt-6-astra"', res)

    def test_idempotent_on_repeated_runs(self) -> None:
        target = 'model = "gpt-5"\nmachine_tag = "local"\n[agents]\nmax_threads = 12\n'
        repo = 'model = "gpt-6-astra"\n[agents]\ndefault_subagent_model = "gpt-6-sol"\nmax_threads = 6\n'
        m1 = merge_toml(repo, target)
        m2 = merge_toml(repo, m1)
        self.assertEqual(m1, m2)

    def test_handles_quoted_table_headers_and_file_paths(self) -> None:
        target = (
            '[hooks.state."/Users/jlaws/.codex/config.toml:user_prompt_submit:0:0"]\n'
            'trusted_hash = "sha256:abc"\n'
            '[projects."/Users/jlaws/Workspace/dotfiles"]\n'
            'trust_level = "trusted"\n'
        )
        repo = 'model = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn('[hooks.state."/Users/jlaws/.codex/config.toml:user_prompt_submit:0:0"]', res)
        self.assertIn('trusted_hash = "sha256:abc"', res)
        self.assertIn('[projects."/Users/jlaws/Workspace/dotfiles"]', res)
        self.assertIn('trust_level = "trusted"', res)
        self.assertIn('model = "gpt-6-astra"', res)

    def test_merges_full_codex_config_preserving_custom_machine_tables(self) -> None:
        repo = (
            'project_doc_fallback_filenames = ["CLAUDE.md", ".agents.md"]\n'
            'model = "gpt-6-astra"\n'
            "[agents]\n"
            'default_subagent_model = "gpt-6-sol"\n'
            'default_subagent_reasoning_effort = "high"\n'
            "max_threads = 6\n"
            "[[hooks.UserPromptSubmit]]\n"
            "[[hooks.UserPromptSubmit.hooks]]\n"
            'type = "command"\n'
            'command = "bash ~/.codex/hooks/log-prompt.sh"\n'
        )
        target = (
            'project_doc_fallback_filenames = ["CLAUDE.md", ".agents.md"]\n'
            'model = "gpt-5"\n'
            "[agents]\n"
            "max_threads = 8\n"
            "# Hook registration comment\n"
            "[[hooks.UserPromptSubmit]]\n"
            "[[hooks.UserPromptSubmit.hooks]]\n"
            'type = "command"\n'
            'command = "bash ~/.codex/hooks/old.sh"\n'
            '[hooks.state."/path/to/config.toml:0:0"]\n'
            'trusted_hash = "sha256:123"\n'
            "[tui]\n"
            "screen_reader_detection_done = true\n"
        )
        res = merge_toml(repo, target)
        self.assertIn('model = "gpt-6-astra"', res)
        self.assertIn("max_threads = 6", res)
        self.assertIn('default_subagent_model = "gpt-6-sol"', res)
        self.assertIn("bash ~/.codex/hooks/log-prompt.sh", res)
        self.assertNotIn("bash ~/.codex/hooks/old.sh", res)
        self.assertIn('[hooks.state."/path/to/config.toml:0:0"]', res)
        self.assertIn('trusted_hash = "sha256:123"', res)
        self.assertIn("[tui]", res)
        self.assertIn("screen_reader_detection_done = true", res)

    def test_raises_on_malformed_section_header(self) -> None:
        with self.assertRaises(ConfigMergeError):
            merge_toml('model = "a"\n', "[unclosed table header\nkey = 1\n")
        with self.assertRaises(ConfigMergeError):
            merge_toml('model = "a"\n', '[table."unclosed quote]\nkey = 1\n')
        with self.assertRaises(ConfigMergeError):
            merge_toml('model = "a"\n', "[table..empty_segment]\nkey = 1\n")
        with self.assertRaises(ConfigMergeError):
            merge_toml('model = "a"\n', "[table.]\nkey = 1\n")
        with self.assertRaises(ConfigMergeError):
            merge_toml('model = "a"\n', "[]\nkey = 1\n")

    def test_raises_on_unclosed_multiline_string(self) -> None:
        with self.assertRaises(ConfigMergeError):
            merge_toml('model = "a"\n', 'text = """unclosed string\n')

    def test_raises_on_duplicate_standard_table(self) -> None:
        with self.assertRaises(ConfigMergeError):
            merge_toml('model = "a"\n', "[agents]\nx = 1\n[agents]\ny = 2\n")


class MergeJsonTests(unittest.TestCase):
    def test_updates_existing_scalar_keys(self) -> None:
        target = '{"outputStyle": "Verbose"}'
        repo = '{"outputStyle": "Concise"}'
        data = json.loads(merge_json(repo, target))
        self.assertEqual(data["outputStyle"], "Concise")

    def test_preserves_target_only_machine_keys(self) -> None:
        target = '{"enabledPlugins": {"my-plugin": true}, "outputStyle": "Verbose"}'
        repo = '{"outputStyle": "Concise", "autoDreamEnabled": true}'
        data = json.loads(merge_json(repo, target))
        self.assertEqual(data["outputStyle"], "Concise")
        self.assertTrue(data["autoDreamEnabled"])
        self.assertTrue(data["enabledPlugins"]["my-plugin"])

    def test_deep_merges_nested_dictionaries(self) -> None:
        target = '{"env": {"MY_TOKEN": "secret", "LIMIT": "10"}}'
        repo = '{"env": {"LIMIT": "20", "ENABLE_FEATURE": "true"}}'
        data = json.loads(merge_json(repo, target))
        self.assertEqual(data["env"]["MY_TOKEN"], "secret")
        self.assertEqual(data["env"]["LIMIT"], "20")
        self.assertEqual(data["env"]["ENABLE_FEATURE"], "true")

    def test_merge_drops_retired_allow_entries_the_target_still_holds(self) -> None:
        target = '{"permissions": {"allow": ["Bash(npx *)", "Bash(ls *)", "Bash(gh *)"]}}'
        repo = '{"permissions": {"allow": ["Bash(git diff *)"]}}'
        data = json.loads(merge_json(repo, target, retired=("Bash(npx *)", "Bash(gh *)")))
        self.assertEqual(data["permissions"]["allow"], ["Bash(git diff *)", "Bash(ls *)"])

    def test_merge_drops_the_legacy_colon_form_of_a_retired_entry(self) -> None:
        target = '{"permissions": {"allow": ["Bash(gh:*)", "Bash(npm test:*)", "Bash(ls *)"]}}'
        repo = '{"permissions": {"allow": ["Bash(git diff *)"]}}'
        retired = ("Bash(gh *)", "Bash(npm test *)")
        data = json.loads(merge_json(repo, target, retired=retired))
        self.assertEqual(data["permissions"]["allow"], ["Bash(git diff *)", "Bash(ls *)"])

    def test_merge_retires_a_colon_form_listed_in_the_retired_set(self) -> None:
        target = '{"permissions": {"allow": ["Bash(gh *)"]}}'
        data = json.loads(merge_json("{}", target, retired=("Bash(gh:*)",)))
        self.assertEqual(data["permissions"]["allow"], [])

    def test_merge_without_a_permissions_key_ignores_retired(self) -> None:
        data = json.loads(merge_json('{"a": 1}', '{"b": 2}', retired=("Bash(gh *)",)))
        self.assertEqual(data, {"a": 1, "b": 2})

    def test_merge_leaves_a_non_list_allow_unchanged(self) -> None:
        target = '{"permissions": {"allow": "Bash(gh *)"}}'
        data = json.loads(merge_json("{}", target, retired=("Bash(gh *)",)))
        self.assertEqual(data["permissions"]["allow"], "Bash(gh *)")

    def test_merge_drops_a_retired_entry_that_comes_from_the_repo(self) -> None:
        repo = '{"permissions": {"allow": ["Bash(gh *)", "Bash(ls *)"]}}'
        target = '{"permissions": {"allow": []}}'
        data = json.loads(merge_json(repo, target, retired=("Bash(gh *)",)))
        self.assertEqual(data["permissions"]["allow"], ["Bash(ls *)"])

    def test_merging_retired_twice_is_a_no_op(self) -> None:
        repo = '{"permissions": {"allow": ["Bash(git diff *)"]}}'
        target = '{"permissions": {"allow": ["Bash(gh:*)", "Bash(ls *)"]}}'
        once = merge_json(repo, target, retired=("Bash(gh *)",))
        self.assertEqual(merge_json(repo, once, retired=("Bash(gh *)",)), once)

    def test_a_bare_string_retired_is_one_entry_not_a_substring_pattern(self) -> None:
        target = '{"permissions": {"allow": ["Bash(gh *)", "Bash(ls *)", "g"]}}'
        data = json.loads(merge_json("{}", target, retired="Bash(gh *)"))
        self.assertEqual(data["permissions"]["allow"], ["Bash(ls *)", "g"])

    def test_unions_permission_lists_without_duplicates(self) -> None:
        target = '{"permissions": {"allow": ["Bash(git diff *)", "Bash(custom)"], "deny": ["Read(~/.secret)"]}}'
        repo = '{"permissions": {"allow": ["Bash(git diff *)", "Bash(git log *)"], "deny": ["Read(~/.ssh/**)"]}}'
        data = json.loads(merge_json(repo, target))
        allow = data["permissions"]["allow"]
        deny = data["permissions"]["deny"]
        self.assertEqual(allow, ["Bash(git diff *)", "Bash(git log *)", "Bash(custom)"])
        self.assertEqual(deny, ["Read(~/.ssh/**)", "Read(~/.secret)"])

    def test_resolves_allow_deny_conflict_with_deny_precedence(self) -> None:
        # If a command appears in deny, it must be removed from allow
        target = '{"permissions": {"allow": ["Bash(dangerous *)"], "deny": []}}'
        repo = '{"permissions": {"allow": [], "deny": ["Bash(dangerous *)"]}}'
        data = json.loads(merge_json(repo, target))
        self.assertNotIn("Bash(dangerous *)", data["permissions"]["allow"])
        self.assertIn("Bash(dangerous *)", data["permissions"]["deny"])

    def test_replaces_hooks_with_repo_hooks(self) -> None:
        target = '{"hooks": {"UserPromptSubmit": [{"command": "old.sh"}]}}'
        repo = '{"hooks": {"UserPromptSubmit": [{"command": "log-prompt.sh"}]}}'
        data = json.loads(merge_json(repo, target))
        self.assertEqual(data["hooks"]["UserPromptSubmit"], [{"command": "log-prompt.sh"}])

    def test_raises_on_malformed_target(self) -> None:
        target = '{"unclosed": '
        repo = '{"outputStyle": "Concise"}'
        with self.assertRaises(ConfigMergeError):
            merge_json(repo, target)

    def test_raises_on_malformed_repo(self) -> None:
        target = '{"outputStyle": "Concise"}'
        repo = '{"unclosed": '
        with self.assertRaises(ConfigMergeError):
            merge_json(repo, target)

    def test_raises_on_non_dict_root(self) -> None:
        target = '["array"]'
        repo = '{"outputStyle": "Concise"}'
        with self.assertRaises(ConfigMergeError):
            merge_json(repo, target)

    def test_raises_on_empty_string(self) -> None:
        with self.assertRaises(ConfigMergeError):
            merge_json('{"key": 1}', "")


class AgentConfigMigrationTests(unittest.TestCase):
    CODEX_REPO = (
        'model = "gpt-6"\n'
        "[agents]\nmax_concurrent_threads_per_session = 6\n"
        "[features.multi_agent_v2]\nenabled = true\nexpose_spawn_agent_model_overrides = true\n"
    )
    CODEX_HOOK = (
        "[[hooks.UserPromptSubmit]]\n"
        "[[hooks.UserPromptSubmit.hooks]]\n"
        'type = "command"\ncommand = "bash ~/.codex/hooks/log-prompt.sh"\n'
    )

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.archive = Archive.create(self.root / "archive", "migration")

    def apply(self, relative: str, repo: str, target: str | None) -> str:
        src = self.root / "repo" / relative
        dest = self.root / "home" / relative
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(repo, encoding="utf-8")
        if target is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(target, encoding="utf-8")
        apply_merged_file(
            src, dest, self.archive, format_type="toml" if relative.endswith(".toml") else "json"
        )
        result = dest.read_text(encoding="utf-8")
        if relative.endswith(".toml") and tomllib is not None:
            tomllib.loads(result)
        return result

    def test_codex_upgrade_removes_retired_hook_and_agent_limits(self) -> None:
        target = (
            "# local settings\n[agents]\nmax_threads = 10\nmax_depth = 3\n"
            "local_flag = true\n" + self.CODEX_HOOK
        )
        result = self.apply(".codex/config.toml", self.CODEX_REPO, target)
        self.assertNotIn("max_threads", result)
        self.assertNotIn("max_depth", result)
        self.assertNotIn("hooks.UserPromptSubmit", result)
        self.assertIn("max_concurrent_threads_per_session = 6", result)
        self.assertIn("# local settings", result)
        self.assertIn("local_flag = true", result)
        backup = self.archive.files_dir / str(self.root / "home/.codex/config.toml").lstrip("/")
        self.assertEqual(backup.read_text(), target)

    def test_codex_preserves_other_handlers_in_same_group_and_other_events(self) -> None:
        target = (
            self.CODEX_HOOK + "# local handler\n[[hooks.UserPromptSubmit.hooks]]\n"
            'type = "command"\ncommand = "bash ~/.codex/hooks/local.sh"\n'
            '[[hooks.Stop]]\n[[hooks.Stop.hooks]]\ntype = "command"\n'
            'command = "bash ~/.codex/hooks/log-prompt.sh"\n'
            '[hooks.state."local"]\ntrusted_hash = "sha256:local"\n'
        )
        result = self.apply(".codex/config.toml", self.CODEX_REPO, target)
        self.assertEqual(result.count("[[hooks.UserPromptSubmit.hooks]]"), 1)
        self.assertIn('command = "bash ~/.codex/hooks/local.sh"', result)
        self.assertIn("# local handler", result)
        self.assertIn("[[hooks.Stop.hooks]]", result)
        self.assertEqual(result.count("bash ~/.codex/hooks/log-prompt.sh"), 1)
        self.assertIn('trusted_hash = "sha256:local"', result)

    def test_codex_preserves_distinct_groups_and_similar_commands(self) -> None:
        target = (
            self.CODEX_HOOK + '[[hooks.UserPromptSubmit]]\nmatcher = "local"\n'
            '[[hooks.UserPromptSubmit.hooks]]\ntype = "command"\n'
            'command = "bash ~/.codex/hooks/log-prompt.sh --local"\n'
            '[[hooks.UserPromptSubmit.hooks]]\ntype = "prompt"\n'
            'command = "bash ~/.codex/hooks/log-prompt.sh"\n'
        )
        result = self.apply(".codex/config.toml", self.CODEX_REPO, target)
        self.assertEqual(result.count("[[hooks.UserPromptSubmit]]"), 1)
        self.assertIn('matcher = "local"', result)
        self.assertIn("log-prompt.sh --local", result)
        self.assertIn('type = "prompt"', result)

    def test_codex_retains_legacy_limits_when_source_has_not_migrated(self) -> None:
        result = self.apply(
            ".codex/config.toml",
            "[agents]\nmax_threads = 6\n",
            "[agents]\nmax_threads = 10\nmax_depth = 3\n",
        )
        self.assertIn("max_threads = 6", result)
        self.assertIn("max_depth = 3", result)

    def test_codex_upgrade_is_idempotent_with_mixed_limits(self) -> None:
        target = (
            "[agents]\nmax_threads = 8\nmax_depth = 3\nmax_concurrent_threads_per_session = 4\n"
        )
        once = self.apply(".codex/config.toml", self.CODEX_REPO, target + self.CODEX_HOOK)
        twice = self.apply(".codex/config.toml", self.CODEX_REPO, once)
        self.assertEqual(once, twice)
        self.assertNotIn("max_threads", twice)
        self.assertNotIn("max_depth", twice)

    def test_codex_promotes_boolean_v2_feature_without_duplicate_table(self) -> None:
        for boolean in ("true", "false"):
            with self.subTest(boolean=boolean):
                target = (
                    'machine = "local"\n[features]\n# keep feature comment\n'
                    f"multi_agent_v2 = {boolean}\nother_feature = false\n"
                )
                once = self.apply(".codex/config.toml", self.CODEX_REPO, target)
                self.assertNotIn(f"multi_agent_v2 = {boolean}", once)
                self.assertIn("[features.multi_agent_v2]\nenabled = true\n", once)
                self.assertIn("other_feature = false", once)
                self.assertIn("# keep feature comment", once)
                self.assertIn('machine = "local"', once)
                self.assertEqual(self.apply(".codex/config.toml", self.CODEX_REPO, once), once)

    def test_codex_promotes_inline_v2_feature_preserving_local_options(self) -> None:
        for key in ("multi_agent_v2", '"multi_agent_v2"', "'multi_agent_v2'"):
            with self.subTest(key=key):
                target = (
                    "[features]\n"
                    f"{key} = {{ enabled = false, expose_spawn_agent_model_overrides = false, "
                    'local_option = "retain, punctuation }", '
                    'local_options = { tags = ["one,two", "three"], enabled = false } }\n'
                    "other_feature = false\n"
                )
                once = self.apply(".codex/config.toml", self.CODEX_REPO, target)
                self.assertNotIn(f"{key} = {{", once)
                self.assertIn("enabled = true", once)
                self.assertIn("expose_spawn_agent_model_overrides = true", once)
                self.assertIn('local_option = "retain, punctuation }"', once)
                self.assertIn(
                    'local_options = { tags = ["one,two", "three"], enabled = false }', once
                )
                self.assertIn("other_feature = false", once)
                self.assertEqual(self.apply(".codex/config.toml", self.CODEX_REPO, once), once)

    def test_codex_normalizes_equivalent_managed_feature_paths(self) -> None:
        targets = (
            '["features"]\nmulti_agent_v2 = false\nother_feature = false\n',
            "['features']\n'multi_agent_v2' = false\nother_feature = false\n",
            "features.multi_agent_v2 = false\nfeatures.other_feature = false\n",
            'features."multi_agent_v2" = { "enabled" = false, local_option = 7 }\n',
            '[features]\nmulti_agent_v2 = { "enabled" = false, local_option = 7 }\n',
            "[features]\nmulti_agent_v2.enabled = false\nmulti_agent_v2.local_option = 7\n",
            "features.multi_agent_v2.enabled = false\nfeatures.multi_agent_v2.local_option = 7\n",
            '["features"."multi_agent_v2"]\n"enabled" = false\nlocal_option = 7\n',
        )
        for target in targets:
            with self.subTest(target=target):
                once = self.apply(".codex/config.toml", self.CODEX_REPO, target)
                self.assertNotIn("= false", once.replace("other_feature = false", ""))
                self.assertIn("enabled = true", once)
                if "local_option" in target:
                    self.assertIn("local_option = 7", once)
                self.assertEqual(self.apply(".codex/config.toml", self.CODEX_REPO, once), once)

    def test_codex_preserves_literal_dots_in_unmanaged_feature_names(self) -> None:
        target = (
            '"features.multi_agent_v2" = false\n'
            '[features]\n"multi_agent_v2.enabled" = false\n'
            'multi_agent_v2 = { "enabled" = false, "local.option" = 7 }\n'
        )
        once = self.apply(".codex/config.toml", self.CODEX_REPO, target)
        self.assertIn('"features.multi_agent_v2" = false', once)
        self.assertIn('"multi_agent_v2.enabled" = false', once)
        self.assertIn('"local.option" = 7', once)
        self.assertIn("enabled = true", once)
        self.assertEqual(self.apply(".codex/config.toml", self.CODEX_REPO, once), once)

    def test_codex_normalizes_quoted_and_dotted_agent_limits(self) -> None:
        targets = (
            '["agents"]\n"max_threads" = 8\n\'max_depth\' = 3\n'
            '"max_concurrent_threads_per_session" = 4\nlocal_option = 7\n',
            "agents.max_threads = 8\nagents.max_depth = 3\n"
            "agents.max_concurrent_threads_per_session = 4\nagents.local_option = 7\n",
        )
        for target in targets:
            with self.subTest(target=target):
                once = self.apply(".codex/config.toml", self.CODEX_REPO, target)
                self.assertNotIn("max_threads", once)
                self.assertNotIn("max_depth", once)
                self.assertIn("max_concurrent_threads_per_session = 6", once)
                self.assertIn("local_option = 7", once)
                self.assertEqual(self.apply(".codex/config.toml", self.CODEX_REPO, once), once)

    def test_codex_keeps_custom_role_and_v2_descendant_tables_in_both_orders(self) -> None:
        roles = '[agents.myrole]\nconfig_file = "mine.toml"\n'
        descendants = "[features.multi_agent_v2.local_options]\npreference = false\n"
        managed = "[agents]\nmax_threads = 2\n[features.multi_agent_v2]\nenabled = false\n"
        for target in (
            roles,
            roles + descendants,
            managed + roles + descendants,
            roles + descendants + managed,
        ):
            with self.subTest(target=target):
                once = self.apply(".codex/config.toml", self.CODEX_REPO, target)
                self.assertIn(roles, once)
                if "local_options" in target:
                    self.assertIn(descendants, once)
                if tomllib is not None:
                    data = tomllib.loads(once)
                    self.assertEqual(data["agents"]["myrole"], {"config_file": "mine.toml"})
                    self.assertEqual(data["agents"]["max_concurrent_threads_per_session"], 6)
                    if "local_options" in target:
                        self.assertEqual(
                            data["features"]["multi_agent_v2"]["local_options"],
                            {"preference": False},
                        )
                self.assertEqual(self.apply(".codex/config.toml", self.CODEX_REPO, once), once)

    def test_codex_expands_only_inline_feature_ancestors_and_keeps_siblings(self) -> None:
        for v2 in ("true", "false", '{ "enabled" = false, local_option = 7 }'):
            with self.subTest(v2=v2):
                target = (
                    f'features = {{ "multi_agent_v2" = {v2}, hooks = true, '
                    "local = { preference = false } } # feature preferences\n"
                    '[agents.myrole]\nconfig_file = "mine.toml"\n'
                )
                once = self.apply(".codex/config.toml", self.CODEX_REPO, target)
                self.assertIn("# feature preferences", once)
                self.assertIn("hooks = true", once)
                self.assertIn("local = { preference = false }", once)
                if tomllib is not None:
                    data = tomllib.loads(once)
                    self.assertEqual(data["features"]["local"], {"preference": False})
                    self.assertIs(data["features"]["hooks"], True)
                    expected: dict[str, bool | int] = {
                        "enabled": True,
                        "expose_spawn_agent_model_overrides": True,
                    }
                    if "local_option" in v2:
                        expected["local_option"] = 7
                    self.assertEqual(data["features"]["multi_agent_v2"], expected)
                    self.assertEqual(data["agents"]["myrole"], {"config_file": "mine.toml"})
                self.assertEqual(self.apply(".codex/config.toml", self.CODEX_REPO, once), once)

    def test_codex_repairs_prior_scalar_and_table_collision_without_losing_local_settings(
        self,
    ) -> None:
        target = (
            'machine = "local"\n[features]\nmulti_agent_v2 = false\nother_feature = false\n'
            "[features.multi_agent_v2]\nenabled = false\nlocal_option = 7\n"
        )
        once = self.apply(".codex/config.toml", self.CODEX_REPO, target)
        self.assertNotIn("multi_agent_v2 = false", once)
        self.assertEqual(once.count("[features.multi_agent_v2]"), 1)
        self.assertIn("local_option = 7", once)
        self.assertIn('machine = "local"', once)
        self.assertIn("other_feature = false", once)
        self.assertEqual(self.apply(".codex/config.toml", self.CODEX_REPO, once), once)

    def test_codex_hook_cleanup_tracks_parents_across_unrelated_tables(self) -> None:
        target = (
            '[[hooks.UserPromptSubmit]]\nmatcher = "mixed"\n'
            '[projects."local"]\ntrust_level = "trusted"\n'
            '[[hooks.UserPromptSubmit.hooks]]\ntype = "command"\n'
            'command = "bash ~/.codex/hooks/log-prompt.sh"\n'
            "[tui]\nlocal_setting = true\n"
            '[[hooks.UserPromptSubmit.hooks]]\ntype = "command"\n'
            'command = "bash ~/.codex/hooks/local.sh"\n'
            '[[hooks.UserPromptSubmit]]\nmatcher = "retired"\n'
            "[features]\nother_feature = true\n"
            '[[hooks.UserPromptSubmit.hooks]]\ntype = "command"\n'
            'command = "bash ~/.codex/hooks/log-prompt.sh"\n'
        )
        result = self.apply(".codex/config.toml", self.CODEX_REPO, target)
        self.assertNotIn("log-prompt.sh", result)
        self.assertEqual(result.count("[[hooks.UserPromptSubmit]]"), 1)
        self.assertIn('matcher = "mixed"', result)
        self.assertNotIn('matcher = "retired"', result)
        self.assertIn('command = "bash ~/.codex/hooks/local.sh"', result)
        self.assertIn('[projects."local"]\ntrust_level = "trusted"', result)
        self.assertIn("[tui]\nlocal_setting = true", result)
        self.assertIn("other_feature = true", result)

    def test_claude_removes_only_retired_handler_from_a_mixed_group(self) -> None:
        old = {"type": "command", "command": "bash ~/.claude/hooks/log-prompt.sh"}
        local = {"type": "command", "command": "bash ~/.claude/hooks/local.sh", "timeout": 4}
        target = json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {"matcher": "*", "hooks": [old, local]},
                        {"hooks": [old]},
                        {"hooks": [{"type": "prompt", "command": old["command"]}]},
                    ],
                    "Stop": [{"hooks": [old]}],
                },
                "machine": "local",
            }
        )
        result = json.loads(self.apply(".claude/settings.json", '{"hooks": {}}', target))
        self.assertEqual(
            result["hooks"]["UserPromptSubmit"],
            [
                {"matcher": "*", "hooks": [local]},
                {"hooks": [{"type": "prompt", "command": old["command"]}]},
            ],
        )
        self.assertEqual(result["hooks"]["Stop"], [{"hooks": [old]}])
        self.assertEqual(result["machine"], "local")

    def test_claude_removes_empty_event_and_is_idempotent(self) -> None:
        target = '{"hooks":{"UserPromptSubmit":[{"hooks":[{"type":"command","command":"bash ~/.claude/hooks/log-prompt.sh"}]}]}}'
        once = self.apply(".claude/settings.json", "{}", target)
        self.assertNotIn("UserPromptSubmit", json.loads(once).get("hooks", {}))
        self.assertEqual(self.apply(".claude/settings.json", "{}", once), once)

    def test_unrecognized_hook_shapes_and_other_config_paths_are_preserved(self) -> None:
        target = '{"hooks":{"UserPromptSubmit":[{"command":"bash ~/.claude/hooks/log-prompt.sh"},"local"]}}'
        result = json.loads(self.apply(".claude/settings.json", "{}", target))
        self.assertEqual(result, json.loads(target))
        for relative in (".claude/settings.local.json", ".gemini/antigravity-cli/settings.json"):
            with self.subTest(relative=relative):
                data = json.loads(self.apply(relative, "{}", target))
                self.assertEqual(data, json.loads(target))
        codex = self.apply(".other/config.toml", self.CODEX_REPO, self.CODEX_HOOK)
        self.assertIn("bash ~/.codex/hooks/log-prompt.sh", codex)

    def test_fresh_install_keeps_source_defaults(self) -> None:
        self.assertEqual(self.apply(".codex/config.toml", self.CODEX_REPO, None), self.CODEX_REPO)
        self.assertEqual(
            self.apply(".claude/settings.json", '{"hooks": {}}\n', None), '{"hooks": {}}\n'
        )


if __name__ == "__main__":
    unittest.main()
