---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "a user hands over a written plan to work through task by task in this session with verification; .claude/skills/executing-plans/SKILL.md; should"
---

Work through the plan below task by task in this session, running each task's verification before moving on.

PLAN: invoice export
- Task 1: Add exportCsv(invoices) in src/export.js, returning a CSV string with a header row. Verify: `node --test test/export.test.js` reports 0 failed.
- Task 2: Add an --export flag to the CLI that writes that CSV to stdout. Verify: `node --test` reports 0 failed.
