---
name: j-debug
description: "Systematic bug investigation — root cause analysis, not random fixes. Use when a test fails, a bug appears, or behavior is unexpected. Do NOT use for asking how to fix something (code directly instead)."
argument-hint: "<bug-description-or-failing-test>"
---

Bug / failing test: $ARGUMENTS

Before investigating, gather diagnostic context:

1. **Check git status** for uncommitted changes (note them for context).
2. **Detect test runner** from project config (package.json scripts, pytest.ini, Makefile, etc.).
3. **Check recent commits** (`git log --oneline -10`) for potential culprits.
4. **Capture failure output**: If $ARGUMENTS references a test name, run it first to get the current failure output.

Then invoke the `debugging-methodology` skill via the Skill tool and apply the Four Phases (root cause investigation → pattern analysis → hypothesis and testing → implementation). When the cause is unclear, use the Structured Hypothesis Investigation section to enumerate and test 3-5 hypotheses; independent hypotheses may be investigated in parallel via subagents.

Once the root cause is found, you may delegate the durable regression test to the `test-writer` agent via the Task tool. Verify it fails before the fix and passes after.
