---
description: "say where each agent works, and send agents that read uncommitted work to the caller's working tree, because a worktree is a fresh checkout without those changes; .claude/skills/dispatching-parallel-agents/SKILL.md; hard because worktree isolation looks like the safe default for parallel helpers and reviews the committed code without any error"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

I have changes to the calc package that I have not committed yet. Have two helpers review them at the same time, one for correctness and one for test coverage; neither may edit anything. Before dispatching, save both briefs to ./briefs.md, and in each brief say where the helper works, what it must not change, and the format it returns. Then merge what they report into ./review.md, one line per finding with file:line.
