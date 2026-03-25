---
name: cmd-workflow-team-investigate
description: "Systematic hypothesis debugging — investigate root causes one theory at a time. Use when debugging complex bugs where the root cause is unclear. Do NOT use for simple bugs (use /debug instead)."
disable-model-invocation: true
---

# Team Investigate

Bug description: the user's provided input

**Do NOT use subagents or parallel agents. Process all hypotheses linearly.**

Use a hypothesis-testing approach to investigate: generate competing theories, then test each one systematically.
