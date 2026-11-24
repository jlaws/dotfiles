---
description: "a user asks to work through PR review comments, fix, reply, and push; .claude/commands/j-pr-fix.md; noauto"
tags: [triggers, trigger-noauto]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Reviewers left three comments on PR #42 for my branch. Go through them, make the fixes, reply to each thread, and push. The comments:
[
  {"id": 11, "reviewer": "ana", "path": "calc/ops.py", "line": 8, "body": "Blocking: this should reject empty input instead of dividing by zero."},
  {"id": 12, "reviewer": "ben", "path": "tests/test_ops.py", "line": 12, "body": "nit: rename test_mean to test_mean_of_three."},
  {"id": 13, "reviewer": "ana", "path": "calc/ops.py", "line": 8, "body": "Question: why not use statistics.mean here?"}
]
