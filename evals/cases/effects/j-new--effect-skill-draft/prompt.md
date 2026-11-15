---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-new Create a new skill for our knowledge base that helps triage noisy application logs. I first thought of the name claude-log-triage. Claude Code cannot write under .claude/ in this session, so write the draft to ./draft/skills/log-triage/SKILL.md instead. Our conventions: a skill name is lowercase letters, numbers and single hyphens and may not contain the words claude or anthropic, so use the closest allowed name in the frontmatter (the folder is already log-triage); the description states triggering conditions only (when to use it), never a summary of the steps, starts with Use when, and has no angle brackets; the frontmatter does not set a model; the body has sections named When to use and Gotchas. Do not add a table of excuses or an all-caps rule list.
