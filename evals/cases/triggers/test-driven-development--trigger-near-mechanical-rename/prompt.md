---
description: "a user wants a pure rename that existing tests already cover, which needs no new test; .claude/skills/test-driven-development/SKILL.md; near: refactoring-and-debt"
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Rename the helper parseCfg to parseConfig across the repo. The existing tests already cover every call site, so I only need the rename done and the suite confirmed green.
