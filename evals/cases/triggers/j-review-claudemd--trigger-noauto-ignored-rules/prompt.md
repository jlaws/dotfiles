---
description: "a user wants recent sessions reviewed for ignored, stale, and missing CLAUDE.md rules; .claude/commands/j-review-claudemd.md; noauto"
tags: [triggers, trigger-noauto]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Look back over my last few sessions with Claude and tell me which rules in my CLAUDE.md it keeps ignoring, which ones are stale, and what I should add.
