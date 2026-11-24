---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "a user just shipped a release with a renamed flag and new config and wants stale docs updated; .claude/skills/post-ship-doc-sync/SKILL.md; should"
---

We just shipped v1.2 with a renamed CLI flag and a new config option. Go through the docs and update whatever is now stale.
