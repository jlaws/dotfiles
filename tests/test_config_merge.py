from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from macos_setup.config_merge import ConfigMergeError, merge_json, merge_toml


class MergeTomlTests(unittest.TestCase):
    def test_updates_existing_scalar_keys(self) -> None:
        target = 'model = "gpt-5"\nmodel_reasoning_effort = "low"\n'
        repo = 'model = "gpt-6-astra"\nmodel_reasoning_effort = "high"\n'
        res = merge_toml(repo, target)
        self.assertIn('model = "gpt-6-astra"', res)
        self.assertIn('model_reasoning_effort = "high"', res)

    def test_preserves_target_only_scalar_keys(self) -> None:
        target = 'machine_name = "work-laptop"\nmodel = "gpt-5"\n'
        repo = 'model = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn('machine_name = "work-laptop"', res)
        self.assertIn('model = "gpt-6-astra"', res)

    def test_merges_tables_and_preserves_target_table_keys(self) -> None:
        target = '[agents]\nmax_threads = 8\nlocal_agent_flag = true\n'
        repo = '[agents]\ndefault_subagent_model = "gpt-6-sol"\nmax_threads = 6\n'
        res = merge_toml(repo, target)
        self.assertIn('[agents]', res)
        self.assertIn('default_subagent_model = "gpt-6-sol"', res)
        self.assertIn('max_threads = 6', res)
        self.assertIn('local_agent_flag = true', res)

    def test_preserves_target_only_sections(self) -> None:
        target = '[mcp_servers.local]\ncommand = "my-mcp"\n'
        repo = 'model = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn('[mcp_servers.local]', res)
        self.assertIn('command = "my-mcp"', res)
        self.assertIn('model = "gpt-6-astra"', res)

    def test_preserves_comments_and_blank_lines(self) -> None:
        target = '# Machine specific comment\nmodel = "gpt-5"\n'
        repo = '# Repo hook comment\nmodel = "gpt-6-astra"\n'
        res = merge_toml(repo, target)
        self.assertIn('# Machine specific comment', res)
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

    def test_idempotent_on_repeated_runs(self) -> None:
        target = 'model = "gpt-5"\nmachine_tag = "local"\n[agents]\nmax_threads = 12\n'
        repo = 'model = "gpt-6-astra"\n[agents]\ndefault_subagent_model = "gpt-6-sol"\nmax_threads = 6\n'
        m1 = merge_toml(repo, target)
        m2 = merge_toml(repo, m1)
        self.assertEqual(m1, m2)

    def test_operates_without_tomllib(self) -> None:
        with patch.dict(sys.modules, {"tomllib": None}):
            res = merge_toml('model = "b"\n', 'model = "a"\n')
            self.assertIn('model = "b"', res)


class MergeJsonTests(unittest.TestCase):
    def test_updates_existing_scalar_keys(self) -> None:
        target = '{"outputStyle": "Verbose"}'
        repo = '{"outputStyle": "Concise"}'
        res = merge_json(repo, target)
        self.assertIn('"outputStyle": "Concise"', res)

    def test_preserves_target_only_machine_keys(self) -> None:
        target = '{"enabledPlugins": {"my-plugin": true}, "outputStyle": "Verbose"}'
        repo = '{"outputStyle": "Concise", "autoDreamEnabled": true}'
        res = merge_json(repo, target)
        self.assertIn('"enabledPlugins"', res)
        self.assertIn('"my-plugin": true', res)
        self.assertIn('"autoDreamEnabled": true', res)
        self.assertIn('"outputStyle": "Concise"', res)

    def test_deep_merges_nested_dictionaries(self) -> None:
        target = '{"env": {"MY_TOKEN": "secret", "LIMIT": "10"}}'
        repo = '{"env": {"LIMIT": "20", "ENABLE_FEATURE": "true"}}'
        res = merge_json(repo, target)
        self.assertIn('"MY_TOKEN": "secret"', res)
        self.assertIn('"LIMIT": "20"', res)
        self.assertIn('"ENABLE_FEATURE": "true"', res)

    def test_unions_permission_lists_without_duplicates(self) -> None:
        target = '{"permissions": {"allow": ["Bash(git diff *)", "Bash(custom)"], "deny": ["Read(~/.secret)"]}}'
        repo = '{"permissions": {"allow": ["Bash(git diff *)", "Bash(git log *)"], "deny": ["Read(~/.ssh/**)"]}}'
        res = merge_json(repo, target)
        self.assertIn('"Bash(git diff *)"', res)
        self.assertIn('"Bash(git log *)"', res)
        self.assertIn('"Bash(custom)"', res)
        self.assertIn('"Read(~/.ssh/**)"', res)
        self.assertIn('"Read(~/.secret)"', res)

    def test_replaces_hooks_with_repo_hooks(self) -> None:
        target = '{"hooks": {"UserPromptSubmit": [{"command": "old.sh"}]}}'
        repo = '{"hooks": {"UserPromptSubmit": [{"command": "log-prompt.sh"}]}}'
        res = merge_json(repo, target)
        self.assertIn('"log-prompt.sh"', res)
        self.assertNotIn('"old.sh"', res)

    def test_raises_on_malformed_target(self) -> None:
        target = '{"unclosed": '
        repo = '{"outputStyle": "Concise"}'
        with self.assertRaises(ConfigMergeError):
            merge_json(repo, target)

    def test_raises_on_non_dict_root(self) -> None:
        target = '["array"]'
        repo = '{"outputStyle": "Concise"}'
        with self.assertRaises(ConfigMergeError):
            merge_json(repo, target)


if __name__ == "__main__":
    unittest.main()
