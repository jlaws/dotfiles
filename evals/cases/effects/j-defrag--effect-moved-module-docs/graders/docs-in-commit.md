---
type: llm
focus: { source: file, path: log.txt }
---

All claims are required.
PASS if log.txt shows README.md modified in the commit that removes calc/fmt_money.py (listed as D, or as the old path of an R rename), or in an earlier commit made after the Initial commit.
FAIL if README.md is modified only in a commit later than the one that removes calc/fmt_money.py, or in no commit after the Initial commit.
FAIL if calc/fmt_money.py or calc/fmt_percent.py is never removed in any commit.
