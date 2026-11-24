---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "require a clean working tree before advancing, and stop without stashing, telling the user to commit or discard first; .claude/commands/j-next.md; hard because a model told the previous part merged tends to stash or carry the stray change along and start the next part"
---

/kb:j-next Continue plans/plan.md. Its first part was merged as PR #7 into origin/main (take that as given; there is no network, so do not try to reach GitHub). Write ./report.md: if you advanced the plan, say what you did; if you could not safely start the next part, say exactly why, what you left as it was, and what I need to do before you can continue. Do not discard any of my work to get going.
