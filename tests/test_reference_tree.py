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


if __name__ == "__main__":
    unittest.main()
