---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-debug the test in tests/test_empty.py fails. Find the root cause before changing anything. First run the failing test and save its failure output to ./before-fix.txt. Then write ./investigation.md with four sections headed Root Cause (give the file and line number as file:line), Hypotheses Tested (at least three, each marked confirmed or refuted with the evidence), Fix, and Verification. Do not edit anything under tests/. Then fix the code, and save the output of the full suite run after the fix to ./after-fix.txt.
