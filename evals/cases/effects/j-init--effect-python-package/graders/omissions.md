---
type: llm
focus: { source: file, path: AGENTS.md }
---

All claims are required.
PASS if AGENTS.md says in a line that no project-level Codex config file was written, with a reason (such as it carries only repo-wide postures that the author should choose).
PASS if AGENTS.md states the test command python3 -m unittest and describes what calc/ holds.
FAIL if AGENTS.md contains generic assistant-behavior advice such as being concise or never claiming success without evidence.
