"""Reachability and integrity checks for the .claude knowledge base.

These wrap the audit script that `skill-audit` runs, so there is one implementation of each
check rather than a test and a script that can drift. Three failures motivated them:

  * every nested reference was linked as `references/<child>.md` when the real path is
    `<parent-stem>/<child>.md`, leaving 24 files unreachable
  * `research/paper-classification.md` was missing from its agent's index
  * three files linked `../SKILL.md#structured-output`, a file that does not exist here
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / ".claude" / "skills" / "skill-audit" / "scripts" / "audit.py"


def load_audit():
    """Import the audit script by path, since its directory is not a package."""
    spec = importlib.util.spec_from_file_location("kb_audit", AUDIT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations via sys.modules, so register before executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReferenceTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit_module = load_audit()

    def run_audit(self):
        """Run every check and return the findings."""
        audit = self.audit_module.Audit(REPO)
        audit.check_skills()
        audit.check_shared_skill_budget()
        audit.check_agents()
        audit.check_commands()
        audit.check_references()
        audit.check_anchors()
        audit.check_config()
        return audit.findings

    def test_audit_script_exists(self):
        self.assertTrue(AUDIT.is_file(), f"missing {AUDIT}")

    def test_no_failing_findings(self):
        fails = [f for f in self.run_audit() if f.severity == self.audit_module.FAIL]
        detail = "\n".join(f"[{f.check}] {f.path}: {f.message}" for f in fails)
        self.assertEqual(fails, [], f"knowledge base has FAIL findings:\n{detail}")

    def test_no_warning_findings(self):
        warns = [f for f in self.run_audit() if f.severity == self.audit_module.WARN]
        detail = "\n".join(f"[{f.check}] {f.path}: {f.message}" for f in warns)
        self.assertEqual(warns, [], f"knowledge base has WARN findings:\n{detail}")

    def test_every_reference_is_reachable(self):
        """No reference may be unreachable from an agent, command, skill, or CLAUDE.md."""
        audit = self.audit_module.Audit(REPO)
        unreachable = sorted(set(audit.reference_stems) - audit._indexed_stems())
        self.assertEqual(unreachable, [], f"unreachable references: {unreachable}")

    def test_relative_reference_paths_resolve(self):
        audit = self.audit_module.Audit(REPO)
        audit.check_references()
        dangling = [f for f in audit.findings if f.check == "XR-7"]
        detail = "\n".join(f"{f.path}: {f.message}" for f in dangling)
        self.assertEqual(dangling, [], f"dangling reference paths:\n{detail}")

    def test_anchors_resolve(self):
        audit = self.audit_module.Audit(REPO)
        audit.check_anchors()
        dangling = [f for f in audit.findings if f.check == "XR-8"]
        detail = "\n".join(f"{f.path}: {f.message}" for f in dangling)
        self.assertEqual(dangling, [], f"dangling anchors:\n{detail}")


class AgentModelPinTests(unittest.TestCase):
    """AG-F8: a pinned model ID fails, a floating alias passes, a missing `model` warns.

    Run against the real tree these would pass even if the check never fired, because every agent
    already declares a floating alias. A synthetic tree is what proves the check does something.
    The rule itself is `CLAUDE.md`, Delegation: an alias survives a model generation, an ID does not.
    """

    @classmethod
    def setUpClass(cls):
        cls.audit_module = load_audit()

    def findings_for(self, frontmatter: str):
        """Audit a throwaway tree holding one agent, and return only its AG-F8 findings."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / ".claude" / "agents").mkdir(parents=True)
        (root / ".claude" / "skills").mkdir(parents=True)
        (root / ".claude" / "agents" / "probe.md").write_text(
            "---\nname: probe\n" + frontmatter + "---\n\n" + "word " * 30 + "\n"
        )
        audit = self.audit_module.Audit(root)
        audit.check_agents()
        return [f for f in audit.findings if f.check == "AG-F8"]

    def test_a_floating_alias_is_accepted(self):
        for alias in ("opus", "sonnet", "haiku", "fable", "inherit"):
            with self.subTest(alias=alias):
                self.assertEqual(self.findings_for("model: " + alias + "\n"), [])

    def test_a_pinned_model_id_fails(self):
        findings = self.findings_for("model: claude-sonnet-4-5-20250929\n")
        self.assertEqual(len(findings), 1, findings)
        self.assertEqual(findings[0].severity, self.audit_module.FAIL)
        self.assertIn("claude-sonnet-4-5-20250929", findings[0].message)

    def test_a_missing_model_warns(self):
        findings = self.findings_for("")
        self.assertEqual(len(findings), 1, findings)
        self.assertEqual(findings[0].severity, self.audit_module.WARN)

    def test_an_empty_model_value_warns_rather_than_reporting_a_pin(self):
        """`frontmatter` stores "" for a bare `model:`, which is not None. Testing only the wholly
        absent key let the empty case report `pins model ``` `` ``` -- the opposite of what happened."""
        for spelling in ("model:\n", "model: \n", 'model: ""\n'):
            with self.subTest(spelling=spelling):
                findings = self.findings_for(spelling)
                self.assertEqual(len(findings), 1, findings)
                self.assertEqual(findings[0].severity, self.audit_module.WARN)
                self.assertIn("declares no `model`", findings[0].message)

    def test_a_mis_cased_alias_names_the_lowercase_fix(self):
        """The generic remedy lists `opus`, so `model: Opus` failed with a message naming a value
        indistinguishable from what the author wrote."""
        findings = self.findings_for("model: Opus\n")
        self.assertEqual(len(findings), 1, findings)
        self.assertEqual(findings[0].severity, self.audit_module.FAIL)
        self.assertIn("must be lowercase `opus`", findings[0].message)

    def test_the_failure_message_carries_the_remedy_not_only_the_defect(self):
        """A finding that names the problem and not the fix costs the reader a lookup."""
        message = self.findings_for("model: claude-sonnet-4-5-20250929\n")[0].message
        for alias in ("opus", "sonnet", "haiku", "fable", "inherit"):
            self.assertIn(alias, message)


class CommandModelTests(unittest.TestCase):
    """CM-F7: a command that sets `model` warns; one that leaves it unset is clean.

    A command runs inside the conversation, so a `model` other than the session's is a model switch
    that re-reads the whole history uncached (docs/adr/workflow/commands-inherit-the-session-model.md).
    """

    @classmethod
    def setUpClass(cls):
        cls.audit_module = load_audit()

    def findings_for(self, frontmatter: str):
        """Audit a throwaway tree holding one command, and return only its CM-F7 findings."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for sub in ("commands", "skills", "agents"):
            (root / ".claude" / sub).mkdir(parents=True)
        (root / ".claude" / "commands" / "j-probe.md").write_text(
            "---\nname: j-probe\n"
            'description: "Probe the audit. Use when testing CM-F7."\n'
            + frontmatter
            + "---\n\n"
            + "word " * 30
            + "\n"
        )
        audit = self.audit_module.Audit(root)
        audit.check_commands()
        return [f for f in audit.findings if f.check == "CM-F7"]

    def test_a_command_without_model_is_clean(self):
        self.assertEqual(self.findings_for("effort: low\n"), [])

    def test_a_command_with_model_warns_and_names_the_value(self):
        """`inherit` warns too: it is a redundant line, and the fix is the same deletion."""
        for value in ("opus", "sonnet", "inherit"):
            with self.subTest(value=value):
                findings = self.findings_for("model: " + value + "\n")
                self.assertEqual(len(findings), 1, findings)
                self.assertEqual(findings[0].severity, self.audit_module.WARN)
                self.assertIn("`model: " + value + "`", findings[0].message)

    def test_a_bare_or_empty_model_still_warns(self):
        """`frontmatter` stores "" for a bare `model:`. CM-F7 tests key presence, unlike AG-F8's
        truthiness, because the remedy for a command is deleting the line either way."""
        for spelling in ("model:\n", "model: \n", 'model: ""\n'):
            with self.subTest(spelling=spelling):
                findings = self.findings_for(spelling)
                self.assertEqual(len(findings), 1, findings)
                self.assertEqual(findings[0].severity, self.audit_module.WARN)


class ToolNameTests(unittest.TestCase):
    """SK-F10 and AG-F7: a current tool name passes, an unknown or retired one warns.

    The real tree declares only valid names, so it cannot show that the check fires. A synthetic
    tree does. `Task` counts as retired: Claude Code renamed it to `Agent` and this tree uses the
    new name, so a stale `Task` should surface rather than pass as an alias.
    """

    @classmethod
    def setUpClass(cls):
        cls.audit_module = load_audit()

    def tree(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for sub in ("agents", "skills", "commands", "references"):
            (root / ".claude" / sub).mkdir(parents=True)
        return root

    def skill_findings(self, tools: str):
        root = self.tree()
        folder = root / ".claude" / "skills" / "probe"
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            "---\nname: probe\ndescription: Use when probing the audit.\n"
            f"allowed-tools: {tools}\n---\n\nBody.\n"
        )
        audit = self.audit_module.Audit(root)
        audit.check_skills()
        return [f.message for f in audit.findings if f.check == "SK-F10"]

    def agent_findings(self, tools: str):
        root = self.tree()
        (root / ".claude" / "agents" / "probe.md").write_text(
            "---\nname: probe\ndescription: Use when probing the audit.\nmodel: sonnet\n"
            f"tools: {tools}\n---\n\n" + "word " * 30 + "\n"
        )
        audit = self.audit_module.Audit(root)
        audit.check_agents()
        return [f.message for f in audit.findings if f.check == "AG-F7"]

    def test_current_tool_names_are_accepted(self):
        for tools in ("Agent", "Skill", "Read, Agent, Bash"):
            with self.subTest(tools=tools):
                self.assertEqual(self.skill_findings(tools), [])
                self.assertEqual(self.agent_findings(tools), [])

    def test_an_unknown_name_warns_and_is_named(self):
        self.assertEqual(
            self.skill_findings("Read, Agent, Bogus"), ["unknown allowed-tools: Bogus"]
        )
        self.assertEqual(self.agent_findings("Read, Bogus"), ["unknown tools: Bogus"])

    def test_the_retired_task_name_warns(self):
        self.assertEqual(self.skill_findings("Task, Read"), ["unknown allowed-tools: Task"])
        self.assertEqual(self.agent_findings("Task"), ["unknown tools: Task"])


class ReferenceCitationTests(unittest.TestCase):
    """XR-7: a citation must name `references/<category>/<file>.md` from the reference root.

    A sibling-relative `references/<file>.md` inside the same category used to pass because the
    check also tried the citing file's own directory. Root `CLAUDE.md` names the root-relative
    form as the one XR-7 validates, so the sibling form is now a finding.
    """

    @classmethod
    def setUpClass(cls):
        cls.audit_module = load_audit()

    def findings_for(self, citation: str):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for sub in ("agents", "skills", "commands"):
            (root / ".claude" / sub).mkdir(parents=True)
        docs = root / ".claude" / "references" / "documentation"
        docs.mkdir(parents=True)
        (docs / "readme-template.md").write_text("# Template\n")
        (docs / "writing.md").write_text(f"# Writing\n\nSee {citation} for the template.\n")
        audit = self.audit_module.Audit(root)
        audit.check_references()
        return [f.message for f in audit.findings if f.check == "XR-7"]

    def test_the_root_relative_form_resolves(self):
        self.assertEqual(self.findings_for("references/documentation/readme-template.md"), [])

    def test_the_sibling_relative_form_is_a_finding(self):
        findings = self.findings_for("references/readme-template.md")
        self.assertEqual(len(findings), 1)
        self.assertIn("actual location references/documentation/readme-template.md", findings[0])

    def test_a_dangling_citation_is_a_finding(self):
        self.assertEqual(len(self.findings_for("references/documentation/missing.md")), 1)


if __name__ == "__main__":
    unittest.main()
