---
description: "a user wants each plan task given to its own fresh agent with a separate reviewer before the next; .claude/skills/subagent-driven-development/SKILL.md; should"
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Run through the plan below, but give each task to its own fresh agent and have every task reviewed by a separate agent before the next one starts.

PLAN: billing migration
- Task 1: Add a currency column to the invoices table with a migration that defaults existing rows to USD. Verify: the migration test passes.
- Task 2: Make invoice totals format with the row's currency instead of a hardcoded dollar sign. Verify: `npm test` reports 0 failed.
