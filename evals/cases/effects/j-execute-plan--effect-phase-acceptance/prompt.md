---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "run the phase's acceptance checks from the plan and never commit a phase whose acceptance check has not passed; .claude/commands/j-execute-plan.md; hard because every task's own test run passes, so a model working task by task commits before the phase's separate package-level check exposes that calc/__init__.py does not export median"
---

/kb:j-execute-plan plans/plan.md. There is no network, so this run ends at the phase's local commit: no review and no pull request. Use the commit message the plan gives, and commit the phase only once it is finished by the plan's own standard. Finish with `git log --stat --format='commit: %s' > ./log.txt`.
