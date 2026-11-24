---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "set up a uv project with a dev dependency, a CI sync that fails on a stale lockfile, uv run for tests, and PEP 723 inline script metadata; .claude/commands/j-lang.md; hard because uv has two lockfile flags for CI and only one of them fails on a stale lockfile"
---

/kb:j-lang set up a Python 3.12 library project with uv and save the instructions to ./setup-notes.md. Include: the commands to create the project and add a runtime dependency (httpx) and a dev dependency (pytest); the exact CI command that installs from the lockfile and fails when the lockfile is out of date; how to run the tests without activating a virtual environment; and a complete single-file script example that declares its own dependency (requests) with inline script metadata and the command to run it.
