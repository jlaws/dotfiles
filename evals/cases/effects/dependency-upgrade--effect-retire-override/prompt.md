---
description: "after an upgrade, remove the temporary overrides and workarounds it retires and keep the ones whose reason still holds; .claude/skills/dependency-upgrade/SKILL.md; hard because the React 17 types pin sits in overrides with its reason in another file, and bumping the version range leaves it in force"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Upgrade this app from React 17 to React 18. You cannot run npm here, so edit the files directly and leave installing to us. Leave package.json in the state you would ship after the upgrade, and keep OVERRIDES.md accurate. Then write ./upgrade-notes.md listing each change you made to package.json with the reason, each entry you kept unchanged in "overrides" with the reason, any code change the upgrade needs, and what we must run and check after installing.
