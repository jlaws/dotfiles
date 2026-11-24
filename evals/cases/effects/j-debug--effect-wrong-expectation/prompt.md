---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "decide where the fault lives before fixing, and fix a wrong test expectation in the test rather than bending the code; .claude/commands/j-debug.md; hard because the request reads like a code bug, and making mean return 2 also turns the suite green"
---

/kb:j-debug tests/test_stats.py fails in this repo and I want the suite green. Find the root cause first, then fix it in the place where the fault actually lives, not just wherever makes the failure disappear. Write ./diagnosis.md stating whether the fault is in the code under test, the test expectation, the test setup, or the environment, with the evidence (run the failing test and quote its output, and show the arithmetic). If you change anything, say which file and why that file.
