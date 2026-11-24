---
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "a user wants PR description text only, to paste in themselves; .claude/commands/j-create-pr.md; near: none"
---

Write me a pull request description for these changes, text only, I'll paste it into GitHub myself: (1) the export now quotes fields that contain commas, (2) added a regression test for quoted fields, (3) bumped the CSV helper's docstring. Use a summary section and a test plan section.
