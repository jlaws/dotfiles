---
type: llm
focus: { source: file, path: CHANGELOG.md }
---

All claims are required.
PASS if the Unreleased section (or a new release section above 0.1.0) records that, without --output, the tool now writes a .jsonl file next to the input instead of printing to stdout.
PASS if the 0.1.0 section still says that without --output the tool prints to stdout, since that was true of that release.
FAIL if the 0.1.0 entry was edited to describe the new behavior or was removed.
