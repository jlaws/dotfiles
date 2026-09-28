from __future__ import annotations

import json
import unittest

from macos_setup.config_merge import ConfigMergeError, merge_json, merge_toml


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
        target = '[agents]\nmax_threads = 8\nlocal_agent_flag = true\n'
        repo = '[agents]\ndefault_subagent_model = "gpt-6-sol"\nmax_threads = 6\n'
        res = merge_toml(repo, target)
        self.assertIn('[agents]', res)
        self.assertIn('default_subagent_model = "gpt-6-sol"', res)
        self.assertIn('max_threads = 6', res)
        self.assertIn('local_agent_flag = true', res)
        self.assertNotIn('max_threads = 8', res)

    def test_preserves_target_only_sections(self) -> None:
        target = '[mcp_servers.local]\ncommand = "my-mcp"\n'
        repo = 'model = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn('[mcp_servers.local]', res)
        self.assertIn('command = "my-mcp"', res)
        self.assertIn('model = "gpt-6-astra"', res)

    def test_appends_repo_only_sections(self) -> None:
        target = 'model = "gpt-5"\n'
        repo = 'model = "gpt-6-astra"\n[agents]\nmax_threads = 4\n'
        res = merge_toml(repo, target)
        self.assertIn('[agents]', res)
        self.assertIn('max_threads = 4', res)

    def test_preserves_comments_and_blank_lines(self) -> None:
        target = '# Machine specific comment\nmodel = "gpt-5"\n'
        repo = '# Repo hook comment\nmodel = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn('# Machine specific comment', res)
        self.assertIn('model = "gpt-6-astra"', res)

    def test_handles_section_headers_with_inline_comments(self) -> None:
        target = '[agents] # host settings\nmax_threads = 8\n'
        repo = '[agents]\ndefault_subagent_model = "gpt-6-sol"\nmax_threads = 4\n'
        res = merge_toml(repo, target)
        self.assertIn('[agents]', res)
        self.assertIn('max_threads = 4', res)
        self.assertIn('default_subagent_model = "gpt-6-sol"', res)
        # Should not duplicate table header
        self.assertEqual(res.count('[agents]'), 1)

    def test_multiline_array_with_quotes_and_comments(self) -> None:
        target = (
            'patterns = [\n'
            '  "[a-z]", # comment with ] bracket\n'
            '  "normal",\n'
            ']\n'
            'model = "gpt-5"\n'
        )
        repo = 'model = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn('"[a-z]"', res)
        self.assertIn('model = "gpt-6-astra"', res)

    def test_multiline_triple_quoted_strings(self) -> None:
        target = 'prompt = """\nhello [\nworld\n"""\nmodel = "gpt-5"\n'
        repo = 'model = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn('hello [', res)
        self.assertIn('model = "gpt-6-astra"', res)

    def test_array_of_tables_hooks_replaced_without_duplication(self) -> None:
        target = (
            '[[hooks.UserPromptSubmit]]\n\n'
            '[[hooks.UserPromptSubmit.hooks]]\n'
            'type = "command"\n'
            'command = "bash ~/.codex/hooks/old.sh"\n'
        )
        repo = (
            '[[hooks.UserPromptSubmit]]\n\n'
            '[[hooks.UserPromptSubmit.hooks]]\n'
            'type = "command"\n'
            'command = "bash ~/.codex/hooks/log-prompt.sh"\n'
        )
        res = merge_toml(repo, target)
        self.assertIn('bash ~/.codex/hooks/log-prompt.sh', res)
        self.assertNotIn('bash ~/.codex/hooks/old.sh', res)

    def test_preserves_target_only_array_of_tables(self) -> None:
        target = (
            '[[custom_plugin.servers]]\n'
            'name = "local"\n'
        )
        repo = 'model = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn('[[custom_plugin.servers]]', res)
        self.assertIn('name = "local"', res)
        self.assertIn('model = "gpt-6-astra"', res)

    def test_idempotent_on_repeated_runs(self) -> None:
        target = 'model = "gpt-5"\nmachine_tag = "local"\n[agents]\nmax_threads = 12\n'
        repo = 'model = "gpt-6-astra"\n[agents]\ndefault_subagent_model = "gpt-6-sol"\nmax_threads = 6\n'
        m1 = merge_toml(repo, target)
        m2 = merge_toml(repo, m1)
        self.assertEqual(m1, m2)

    def test_raises_on_malformed_section_header(self) -> None:
        with self.assertRaises(ConfigMergeError):
            merge_toml('model = "a"\n', '[unclosed table header\nkey = 1\n')

    def test_raises_on_unclosed_multiline_string(self) -> None:
        with self.assertRaises(ConfigMergeError):
            merge_toml('model = "a"\n', 'text = """unclosed string\n')

    def test_raises_on_duplicate_standard_table(self) -> None:
        with self.assertRaises(ConfigMergeError):
            merge_toml('model = "a"\n', '[agents]\nx = 1\n[agents]\ny = 2\n')


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
        self.assertEqual(
            data["hooks"]["UserPromptSubmit"], [{"command": "log-prompt.sh"}]
        )

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


if __name__ == "__main__":
    unittest.main()
