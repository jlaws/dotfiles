---
description: "compare parsed values or whole tokens in assertions, not a substring that also appears in an unrelated line; .claude/skills/test-driven-development/SKILL.md; hard because every report starts with a Checked fields line that already names timeout_ms, so a contains-timeout_ms assertion passes with the check missing"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

validate() in src/report.js checks retries but not timeout_ms. Make it also reject a timeout_ms that is not a positive integer, so that formatReport(config, validate(config)) reports an error for timeout_ms for a config such as { retries: 3, timeout_ms: -1 }. Write the test first, in test/report.test.js, and keep the existing tests there as they are. Every assertion you add must fail if the behavior it names breaks. Run node --test before you finish.
