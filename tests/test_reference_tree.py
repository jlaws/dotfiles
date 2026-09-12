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


if __name__ == "__main__":
    unittest.main()
