"""Architecture checks for the tool-specific agent configuration trees."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FRONTMATTER_DESCRIPTION = re.compile(r'^description:\s*["\']?(.*?)["\']?$', re.MULTILINE)

# Asset-set parity is enforced by name; bodies and descriptions are free to diverge per tool.
# `.claude/` is rightsized for the Claude 5 generation, so its wording no longer tracks `.agents/`.
# Anything that legitimately exists in only one tree is declared here rather than tolerated
# silently, so an accidental deletion still fails.

# Claude-native commands with no shared `cmd-j-*` workflow behind them.
CLAUDE_ONLY_COMMANDS: frozenset[str] = frozenset()

# Shared workflows deliberately absent from `.claude/skills/`.
SHARED_ONLY_SKILLS: frozenset[str] = frozenset()

# Skills that exist only for Claude, and agents that exist in only one tree.
CLAUDE_ONLY_SKILLS: frozenset[str] = frozenset()
CLAUDE_ONLY_AGENTS: frozenset[str] = frozenset()

J_PLAN_COMMANDS = (
    REPO / ".agents" / "skills" / "cmd-j-plan" / "SKILL.md",
    REPO / ".codex" / "prompts" / "j-plan.md",
    REPO / ".gemini" / "commands" / "j-plan.toml",
)

PLAN_STORAGE_SKILLS = (
    REPO / ".agents" / "skills" / "writing-plans" / "SKILL.md",
)

PLAN_EXECUTION_CONSUMERS = (
    REPO / ".agents" / "skills" / "cmd-j-execute-plan" / "SKILL.md",
    REPO / ".agents" / "skills" / "cmd-j-next" / "SKILL.md",
    REPO / ".codex" / "prompts" / "j-execute-plan.md",
    REPO / ".codex" / "prompts" / "j-next.md",
    REPO / ".gemini" / "commands" / "j-execute-plan.toml",
    REPO / ".gemini" / "commands" / "j-next.toml",
)

ACTIVE_PLAN_CONSUMERS = (
    REPO / ".agents" / "skills" / "cmd-j-diff-review" / "SKILL.md",
    REPO / ".codex" / "prompts" / "j-diff-review.md",
    REPO / ".gemini" / "commands" / "j-diff-review.toml",
)

# A worktree carries no uncommitted changes and branches from a start point the agent did not pick,
# so a review run in one silently reviews the wrong commit. Each statement below has one owner --
# repeating it across surfaces is what `writing-skills` warns against -- and the owners are pinned so
# the guidance cannot quietly disappear from a tree.

# Owns "which workspace does a dispatched agent get".
WORKSPACE_SELECTION_OWNERS = (
    REPO / ".claude" / "skills" / "dispatching-parallel-agents" / "SKILL.md",
    REPO / ".agents" / "skills" / "dispatching-parallel-agents" / "SKILL.md",
)

# Owns "which commit does a worktree start from, and how do you confirm it".
WORKTREE_BASE_OWNERS = (
    REPO / ".claude" / "skills" / "using-git-worktrees" / "SKILL.md",
    REPO / ".agents" / "skills" / "using-git-worktrees" / "SKILL.md",
)

# Always-loaded configs carry a pointer, not a restatement.
WORKTREE_POINTERS = (
    REPO / ".claude" / "CLAUDE.md",
    REPO / ".codex" / "AGENTS.md",
    REPO / ".gemini" / "GEMINI.md",
)

# Review dispatch hands each agent the HEAD SHA it should be sitting on.
DIFF_REVIEW_DISPATCHERS = (
    REPO / ".claude" / "commands" / "j-diff-review.md",
    REPO / ".agents" / "skills" / "cmd-j-diff-review" / "SKILL.md",
    REPO / ".codex" / "prompts" / "j-diff-review.md",
    REPO / ".gemini" / "commands" / "j-diff-review.toml",
)


def skill_directories(root: Path) -> set[str]:
    return {path.parent.name for path in root.glob("*/SKILL.md")}


def description(path: Path) -> str:
    match = FRONTMATTER_DESCRIPTION.search(path.read_text())
    if match is None:
        raise AssertionError(f"missing description: {path}")
    return match.group(1)


class AgentConfigArchitectureTests(unittest.TestCase):
    def test_shared_skills_do_not_contain_agent_wrappers(self):
        wrappers = sorted((REPO / ".agents" / "skills").glob("agent-*/SKILL.md"))
        self.assertEqual(wrappers, [])

    def test_command_skills_match_all_native_command_sets(self):
        command_skills = {
            name.removeprefix("cmd-")
            for name in skill_directories(REPO / ".agents" / "skills")
            if name.startswith("cmd-j-")
        }
        codex = {
            path.stem for path in (REPO / ".codex" / "prompts").glob("j-*.md")
        }
        claude = {
            path.stem for path in (REPO / ".claude" / "commands").glob("j-*.md")
        }
        gemini = {
            path.stem for path in (REPO / ".gemini" / "commands").glob("j-*.toml")
        }

        self.assertLessEqual(command_skills, codex)
        self.assertLessEqual(command_skills, gemini)
        # Claude is compared exactly so a deleted command fails instead of passing as a subset.
        self.assertEqual(command_skills, claude - CLAUDE_ONLY_COMMANDS)

    def test_native_agent_sets_match(self):
        codex = {
            path.stem for path in (REPO / ".codex" / "agents").glob("*.toml")
        }
        claude = {
            path.stem for path in (REPO / ".claude" / "agents").glob("*.md")
        }
        gemini = {
            path.stem for path in (REPO / ".gemini" / "agents").glob("*.md")
        }

        self.assertEqual(codex, claude - CLAUDE_ONLY_AGENTS)
        self.assertEqual(codex, gemini)

    def test_shared_skill_descriptions_fit_codex_budget(self):
        for skill in (REPO / ".agents" / "skills").glob("*/SKILL.md"):
            value = description(skill)
            with self.subTest(skill=skill.parent.name):
                self.assertLessEqual(len(value), 64)
                self.assertTrue(value.startswith("Use when"))

    def test_workflow_skill_sets_match_claude_mirror(self):
        """Shared workflows and their Claude counterparts must cover the same set of names.

        Bodies and descriptions are intentionally allowed to diverge: `.claude/` is written for
        the Claude 5 generation while `.agents/` serves Codex and Gemini. Only membership is
        pinned, so a skill cannot silently vanish from one tree.
        """
        shared = {
            name
            for name in skill_directories(REPO / ".agents" / "skills")
            if not name.startswith("cmd-j-")
        }
        claude = skill_directories(REPO / ".claude" / "skills")

        self.assertEqual(shared - SHARED_ONLY_SKILLS, claude - CLAUDE_ONLY_SKILLS)

    def test_parity_exceptions_are_live(self):
        """Every declared exception must still exist, so stale entries cannot hide a real gap."""
        for name in CLAUDE_ONLY_COMMANDS:
            with self.subTest(command=name):
                self.assertTrue((REPO / ".claude" / "commands" / f"{name}.md").is_file())
        for name in SHARED_ONLY_SKILLS:
            with self.subTest(skill=name):
                self.assertTrue((REPO / ".agents" / "skills" / name / "SKILL.md").is_file())
                self.assertFalse((REPO / ".claude" / "skills" / name).exists())
        for name in CLAUDE_ONLY_SKILLS:
            with self.subTest(skill=name):
                self.assertTrue((REPO / ".claude" / "skills" / name / "SKILL.md").is_file())
        for name in CLAUDE_ONLY_AGENTS:
            with self.subTest(agent=name):
                self.assertTrue((REPO / ".claude" / "agents" / f"{name}.md").is_file())

    def test_workspace_selection_is_documented_where_dispatch_is_decided(self):
        for path in WORKSPACE_SELECTION_OWNERS:
            content = path.read_text().lower()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("## workspace selection", content)
                self.assertIn("uncommitted", content)
                self.assertIn("git worktree list", content)

    def test_worktree_skills_pin_a_start_point_and_verify_on_entry(self):
        for path in WORKTREE_BASE_OWNERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("start point", content.lower())
                self.assertIn("git rev-parse HEAD", content)
                # `git worktree add <path> -b <branch>` with no trailing ref uses the current HEAD.
                bare = [
                    line
                    for line in content.splitlines()
                    if line.strip().startswith("git worktree add")
                    and len(line.split("-b", 1)[1].split()) < 2
                ]
                self.assertEqual(bare, [], f"{path.name}: `git worktree add` with no start point")

    def test_always_loaded_configs_point_at_the_workspace_rule(self):
        for path in WORKTREE_POINTERS:
            content = path.read_text().lower()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("worktree", content)
                self.assertIn("uncommitted", content)
                self.assertIn("dispatching-parallel-agents", content)

    def test_diff_review_hands_reviewers_the_head_sha(self):
        for path in DIFF_REVIEW_DISPATCHERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("git rev-parse HEAD", content)
                self.assertIn("HEAD SHA", content)
                self.assertIn("do not create or request a worktree", content)

    def test_worktree_base_ref_is_pinned_to_local_head(self):
        """`fresh`, the harness default, branches agent-isolation worktrees off origin/<default>."""
        settings = json.loads((REPO / ".claude" / "settings.json").read_text())
        self.assertEqual(settings.get("worktree", {}).get("baseRef"), "head")

    def test_cursor_configuration_is_not_tracked(self):
        cursor = REPO / ".cursor"
        self.assertFalse(cursor.exists() and any(path.is_file() for path in cursor.rglob("*")))

    def test_j_plan_commands_persist_the_plan_outside_git(self):
        ignore_patterns = {
            line.strip()
            for line in (REPO / ".gitignore").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("scratchpad/", ignore_patterns)

        for path in J_PLAN_COMMANDS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("scratchpad/plans/", content)
                self.assertIn("git check-ignore -q scratchpad/", content)
                self.assertIn("${TMPDIR:-/tmp}/j-plan/<repo-id>/", content)
                self.assertIn("stable SHA-256 digest", content)
                self.assertIn("Create the parent directory and file immediately", content)
                self.assertIn("0700", content)
                self.assertIn("0600", content)
                self.assertRegex(content, r"exclusive,\s+no-clobber")
                self.assertIn("numeric suffix", content)
                self.assertRegex(content, r"owned by (?:the )?current user")
                self.assertIn("not symlinks", content)
                self.assertIn("Status: Researching", content)
                self.assertIn("## Planning Notes", content)
                self.assertRegex(content, r"source\s+of truth")
                self.assertIn("MUST NOT keep the only copy in context", content)

    def test_plan_storage_skills_forbid_context_only_plans(self):
        for path in PLAN_STORAGE_SKILLS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("scratchpad/plans/", content)
                self.assertIn("git check-ignore -q scratchpad/", content)
                self.assertIn("${TMPDIR:-/tmp}/j-plan/<repo-id>/", content)
                self.assertIn("0700", content)
                self.assertIn("0600", content)
                self.assertRegex(content, r"exclusive,\s+no-clobber")
                self.assertRegex(content, r"owned by (?:the )?current user")
                self.assertIn("not symlinks", content)
                self.assertIn("MUST NOT keep the only copy in context", content)

    def test_plan_execution_commands_confirm_discovered_files(self):
        for path in PLAN_EXECUTION_CONSUMERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("scratchpad/plans/", content)
                self.assertIn("${TMPDIR:-/tmp}/j-plan/<repo-id>/", content)
                self.assertIn("regular, non-symlink plan files", content)
                self.assertIn("full paths", content)
                self.assertIn("modification times", content)
                self.assertIn("even when there is only one", content)
                self.assertIn("MUST NOT execute a discovered plan without confirmation", content)

    def test_active_plan_commands_use_the_persisted_locations(self):
        for path in ACTIVE_PLAN_CONSUMERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("scratchpad/plans/", content)
                self.assertIn("${TMPDIR:-/tmp}/j-plan/<repo-id>/", content)


if __name__ == "__main__":
    unittest.main()
