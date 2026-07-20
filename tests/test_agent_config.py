"""Architecture checks for the tool-specific agent configuration trees."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FRONTMATTER_DESCRIPTION = re.compile(r'^description:\s*["\']?(.*?)["\']?$', re.MULTILINE)


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
        self.assertLessEqual(command_skills, claude)
        self.assertLessEqual(command_skills, gemini)

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

        self.assertEqual(codex, claude)
        self.assertEqual(codex, gemini)

    def test_shared_skill_descriptions_fit_codex_budget(self):
        for skill in (REPO / ".agents" / "skills").glob("*/SKILL.md"):
            value = description(skill)
            with self.subTest(skill=skill.parent.name):
                self.assertLessEqual(len(value), 64)
                self.assertTrue(value.startswith("Use when"))

    def test_workflow_descriptions_match_claude_mirror(self):
        shared_root = REPO / ".agents" / "skills"
        claude_root = REPO / ".claude" / "skills"
        for name in skill_directories(claude_root):
            with self.subTest(skill=name):
                self.assertEqual(
                    description(shared_root / name / "SKILL.md"),
                    description(claude_root / name / "SKILL.md"),
                )

    def test_cursor_configuration_is_not_tracked(self):
        cursor = REPO / ".cursor"
        self.assertFalse(cursor.exists() and any(path.is_file() for path in cursor.rglob("*")))


if __name__ == "__main__":
    unittest.main()
