---
type: llm
focus: { source: file, path: retro-since-v1.md }
---

All claims are required.
PASS if retro-since-v1.md states 1 merge commit since v1.0.0.
PASS if retro-since-v1.md lists the revert of "feat: add power" and lists the branch feat/median as unmerged with 2 commits ahead of main.
PASS if the top-files table shows calc/ops.py touched by 6 commits and README.md by 2 (counting only commits on main since the tag), with churn worded as potential instability.
FAIL if retro-since-v1.md omits the unmerged branch or contains a per-author breakdown.
