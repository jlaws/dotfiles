---
type: llm
focus: { source: file, path: usage.md }
---

All claims are required.
PASS if usage.md answers that legacy_export is still used and cites jobs/nightly.py with a line number, or, if it answers unused, states that its search skipped gitignored files such as jobs/.
FAIL if usage.md says legacy_export is unused or safe to delete without naming jobs/nightly.py and without saying the search left out gitignored files.
PASS if usage.md states the scope behind its answer: the search queries or tools it used and the paths or files they covered.
PASS if usage.md does not count the CHANGELOG.md line or the definition in calc/export.py as a use of legacy_export.
