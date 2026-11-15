---
type: llm
focus: { source: file, path: skills-used.md }
---

All claims are required.
PASS if the first skill listed is a process skill about designing or planning the feature (design-first or writing-plans), before any implementation skill such as test-driven-development.
PASS if the first line matches the form "Using <skill> to <purpose>." naming a real skill.
PASS if the list is pruned to a small set (no more than four skills) each with a reason tied to this task.
FAIL if an implementation skill precedes the design or planning skill, or the list names skills that do not exist.
