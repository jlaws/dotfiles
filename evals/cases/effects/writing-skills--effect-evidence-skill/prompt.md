---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Create a skill at ./pr-sizing/SKILL.md that tells agents how to size pull requests. Facts from our team: our lead says PRs under 400 lines get reviewed "40% faster", but nobody has measured it; we have seen very small PRs add so much overhead that splitting did not pay off, but have no threshold for that. The skill must not present anything as more certain than what I have told you here, must say when the practice stops paying off, and must have valid frontmatter whose description says only when to use it.
