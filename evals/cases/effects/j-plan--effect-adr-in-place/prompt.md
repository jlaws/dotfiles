---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "when an existing ADR covers the decision a plan changes, add a task that updates that ADR in place, revising its Decision, ruling out the previous approach, and bumping updated, rather than adding a second file; .claude/commands/j-plan.md; hard because the spec never mentions the decision record, so a plan aimed at ops.py and its tests leaves the ADR asserting the old behavior or adds a new ADR beside it"
---

/kb:j-plan Callers of calc now need to tell an empty input from a real zero mean. Change mean([]) in calc/ops.py to raise ValueError instead of returning 0.0, test-first. Plan mode is not available in this session, so use ./plans/plan.md as the plan file in its place; plans/ is git-ignored here, so the plan stays out of the repository. The plan must cover every change the repository needs to stay consistent with the new behavior, name each file it touches, and give python3 -m unittest as the verification command.
