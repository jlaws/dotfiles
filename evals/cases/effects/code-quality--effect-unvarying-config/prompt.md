---
description: "an option with one value at every call site and an abstraction with one implementation are anti-patterns, while a TODO with a ticket and reason is acceptable; .claude/skills/code-quality/SKILL.md; hard because a defaulted retries parameter looks like good flexibility until every call site is read"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Review the app/ package for code that should not exist or could be simpler, judging each definition by how the rest of the package actually uses it. Save ./quality-review.md with one line per finding formatted `file:line - problem - fix`, then one line starting `Sanctioned: ` that lists anything you looked at closely and judged acceptable as it is. Do not edit app/.
