---
description: "a skill description states only triggering conditions, in one short line, never the workflow; .claude/skills/writing-skills/SKILL.md; hard because the obvious repair summarizes what the skill does"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

The skill at .claude/skills/broken-skill/SKILL.md is not loading properly. Write the fixed version to ./broken-skill/SKILL.md (files under .claude/ cannot be written here) so it is a good skill: it must have valid frontmatter, and the description is what decides when the skill is picked up, so write it so the skill fires on the right requests, and keep it to one short line. Keep the skill name. Do not change any other file.
