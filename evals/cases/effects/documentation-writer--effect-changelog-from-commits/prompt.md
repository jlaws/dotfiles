---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:documentation-writer agent to write the release notes for version 2.0.0 of the `widgetkit` client library from the change list in ./changes.txt. Save them to ./CHANGELOG.md in Keep a Changelog style with the date 2026-11-15, and group entries by category. Requirements: leave out internal noise that readers of the library do not care about (CI, dependency bumps, merge commits, refactors with no behavior change); give breaking changes their own Breaking section with a Migration section of concrete before-and-after steps; and keep the issue numbers as references.
