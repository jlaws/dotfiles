---
type: llm
focus: { source: file, path: docs/adr/cli/output-flag-name.md }
---

All claims are required.
PASS if the Decision section now states that the flag naming the output file is --output.
PASS if the Ruled Out table has a row for --out with a reason and the date 2026-11-24, and keeps the existing -o row.
PASS if the frontmatter has updated: 2026-11-24 and still has created: 2026-03-02.
FAIL if the Decision section still names --out as the chosen flag, or if the change is only an appended note that leaves the old Decision text in place.
