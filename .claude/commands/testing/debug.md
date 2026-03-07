---
name: debug
description: "Systematic bug investigation — root cause analysis, not random fixes. Use when a test fails, a bug appears, or behavior is unexpected. Do NOT use for asking how to fix something (code directly instead)."
argument-hint: "<bug-description-or-failing-test>"
---

Before invoking the skill, gather diagnostic context:

1. **Check git status** for uncommitted changes (note them for context).
2. **Detect test runner** from project config (package.json scripts, pytest.ini, Makefile, etc.).
3. **Check recent commits** (`git log --oneline -10`) for potential culprits.
4. **Capture failure output**: If $ARGUMENTS references a test name, run it first to get the current failure output.

Invoke the testing:debugging-methodology skill and use it to investigate: $ARGUMENTS
