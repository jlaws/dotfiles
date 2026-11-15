---
type: llm
focus: { source: file, path: bib-report.md }
---

All claims are required.
PASS if zheng2023judging and zheng2023mtbench are identified as duplicates despite different keys and different title casing, with a proposed keeper.
PASS if wang2023selfconsistency is flagged as missing its year.
FAIL if the report treats any pair other than the zheng pair as duplicates.
FAIL if the report says it deleted or rewrote entries instead of proposing merges.
