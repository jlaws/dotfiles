---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "fix README flag drift against the code and file the correction in a Keep a Changelog file under its accurate category, with an Unreleased section and ISO dates; .claude/commands/j-docs.md; hard because a renamed flag reads like a breaking change, but the documented flag never worked"
---

/kb:j-docs the README for tool.py has drifted from the code. Fix ./README.md so every flag it documents matches what tool.py really accepts, then create ./CHANGELOG.md for the project's first release, version 1.0.0 dated 2026-11-14. Requirements for the changelog: Keep a Changelog layout with an Unreleased section above the release, entries filed under the right category heading, and ISO dates. Record the README correction as a changelog entry only if it belongs there, and put it under whichever category is accurate for what changed.
