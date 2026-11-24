---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "in a published library, give dependencies lower bounds and add an upper cap only at a version known to break the library; .claude/references/languages/python-packaging-and-distribution.md; hard because a below-1.0 cap on a 0.x dependency looks prudent, while the one cap backed by a known break has to stay"
---

/kb:j-lang we publish the library in ./pyproject.toml to PyPI next week, and ./docs/compat.md records what we know about each dependency. Set the dependency version constraints the way they should be for a published library, edit ./pyproject.toml in place, and give the reason for each constraint in ./deps-notes.md.
