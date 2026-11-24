---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "under reduced motion remove translation, scaling, and autoplaying loops but keep short opacity and color changes that confirm an action or show state, instead of a global near-zero duration; .claude/references/frontend/web-animation-patterns.md; hard because the global 0.01ms reset is the widely copied reduced-motion snippet and the seeded file already uses it"
---

Use the kb:frontend-engineer agent to correct how ./src/motion.css treats users who have turned on reduced motion in their operating system. Save the corrected stylesheet to ./src/motion.fixed.css (leave src/motion.css unchanged), and save ./motion-notes.md with one line per animation or transition in the file saying what it does under reduced motion.
