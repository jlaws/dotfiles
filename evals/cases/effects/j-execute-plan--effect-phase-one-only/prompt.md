---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "execute only the current PR boundary, commit once per phase with the plan's message, and record the SHA in the Progress ledger without rewriting the plan; .claude/commands/j-execute-plan.md; hard because the next phase is a one-line README edit, so a model tends to finish it too or tidy the plan"
---

/kb:j-execute-plan Carry out only Phase 1 of plans/plan.md today. Phase 2 ships in a later pull request, so leave it, its tasks, and its checklist entry untouched. There is no network, so this run ends at the phase's local commit: no review and no pull request. The new test goes in tests/test_empty.py as the plan says. Use the commit message the plan gives. After committing, tick Phase 1 in the plan's Progress checklist with that commit's SHA and leave that edit uncommitted. Finish with `git log --oneline > ./log.txt`.
