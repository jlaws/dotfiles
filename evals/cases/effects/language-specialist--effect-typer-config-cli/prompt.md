---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "write a Typer CLI with no_args_is_help, Annotated parameters, a missing-key error on stderr with exit status 1, a platformdirs config path, and TOML written with a writer package after merging existing keys; .claude/agents/language-specialist.md; hard because tomllib cannot write TOML, and a hand-built config path misses one of the two OS conventions"
---

Use the kb:language-specialist agent to write a Typer command-line tool for Python 3.12 and save it as ./mytool.py, plus ./scripts-snippet.toml holding the [project.scripts] table. Requirements: running mytool with no arguments prints the help text; `mytool config set KEY VALUE` writes the pair into a config.toml file in the user's configuration directory for the current OS (it must work on both macOS and Linux conventions) and keeps keys already in the file; `mytool config get KEY` prints the value, or prints an error to stderr and exits with status 1 when the key is missing; use typing.Annotated for options and arguments; name the root Typer object app, and point the console script entry point at that object itself, not a wrapper function. The standard library cannot write TOML, so say in a comment which package you use for that.
