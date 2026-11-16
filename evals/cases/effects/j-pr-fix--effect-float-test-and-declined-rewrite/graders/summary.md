---
type: llm
focus: { source: file, path: summary.md }
---

All claims are required.
PASS if summary.md lists comment 603 (the numpy rewrite) under Deferred Items with a reason, and shows no code change for it.
PASS if the Verification section quotes the observed unittest result line (for example Ran N tests ... OK).
FAIL if summary.md says comment 603 was implemented or that a dependency was added.
