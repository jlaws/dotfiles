# Task: Audit Default Agent Tool Permissions

## Context
During diff review of `feat/merge-agent-configs`, security review identified that default configuration files in the repository contain broad tool permissions:
- `.gemini/antigravity-cli/settings.json`: `command(python3)`, `command(python)`, `command(make)`, `command(find)`, `command(xargs)`, `command(chmod)`
- `.claude/settings.json`: `Bash(npx *)`, `Bash(chmod *)`, `Bash(xargs *)`

## Problem
Allowing execution wrappers (`python3 -c`, `find -exec`, `xargs`) without confirmation can permit unintended shell execution if an agent processes untrusted inputs.

## Scope
1. Audit necessity of each broad wrapper command in `.gemini/antigravity-cli/settings.json` and `.claude/settings.json`.
2. Restrict to specific subcommands or scripts where possible.
3. Verify impact on developer workflows and test suites.
