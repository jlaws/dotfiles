---
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "reviewing a branch diff as a reviewer with notes only, not resolving comments; .claude/skills/pr-comment-resolution/SKILL.md; near: code-review-patterns"
---

Look at the diff in this branch against main as a reviewer and tell me what you would flag. I only want your review notes, not changes.
