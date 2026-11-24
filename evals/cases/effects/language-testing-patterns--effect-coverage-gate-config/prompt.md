---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "a coverage gate measures shipped code only, with branch coverage on, test files and generated code and TYPE_CHECKING blocks excluded, at an 80 percent threshold; .claude/skills/language-testing-patterns/SKILL.md; hard because a default config counts tests and generated files, misses the TYPE_CHECKING block, and measures lines only, while a small helper module looks excludable but is shipped logic"
---

Add coverage.py configuration for this repo in ./.coveragerc, with a pass/fail threshold that CI can enforce. Do not install anything and do not add CI files.

Decide what the gate measures. It should count only code we wrote and ship, catch an untested side of an if even when every line ran, and leave out lines that can never run under test. Choose the threshold. Then write ./coverage-notes.md explaining each setting you chose and why.
