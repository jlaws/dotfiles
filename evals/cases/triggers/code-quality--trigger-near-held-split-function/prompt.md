---
description: "a request to split one long function and flatten its nesting, a hands-on refactor rather than a quality assessment; .claude/skills/code-quality/SKILL.md; near: refactoring-and-debt"
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

process_orders in billing.py is 300 lines with deeply nested conditionals. Split it into smaller functions and flatten the nesting, keeping behaviour identical.
