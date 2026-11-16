---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:language-specialist agent to write a Typer command-line tool for Python 3.12 and save it as ./mytool.py, plus ./scripts-snippet.toml holding the [project.scripts] table. Requirements: running mytool with no arguments prints the help text; `mytool config set KEY VALUE` writes the pair into a config.toml file in the user's configuration directory for the current OS (it must work on both macOS and Linux conventions) and keeps keys already in the file; `mytool config get KEY` prints the value, or prints an error to stderr and exits with status 1 when the key is missing; use typing.Annotated for options and arguments; the console script entry point must point at the Typer object itself, not a wrapper function. The standard library cannot write TOML, so say in a comment which package you use for that.
