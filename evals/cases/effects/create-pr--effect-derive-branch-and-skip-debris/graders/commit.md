---
type: llm
focus: { source: file, path: log.txt }
---

All claims are required.
PASS if log.txt shows exactly one commit beyond "Initial commit", whose subject is in imperative mood, under 72 characters, and has no trailing period.
FAIL if log.txt shows more than one new commit or a subject like "Update files" with no description of the change.
