---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "a user hands over requirements and asks for a test-first implementation plan with paths, verification, and PR splits; .claude/commands/j-plan.md; should"
---

Here are the requirements for invoice reminder emails: send one reminder 3 days before the due date and one the day after it is overdue, skip invoices already paid, and log each send. Turn that into an implementation plan for our Python service with test-first steps, exact file paths, how we verify each phase, and how it splits into pull requests.
