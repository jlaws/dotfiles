---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Scaffold a new Python command-line project in this empty directory: a tool called mdlite that converts a markdown file to HTML. It should depend on a markdown library, use a linter and a test runner, and have pinned dependency versions in pyproject.toml. There is no network access in this session. Do not install anything. Leave CI workflows and dependency-update bots out of this scaffold.

Write the project files and the CLAUDE.md, AGENTS.md, and GEMINI.md config. End with ./SCAFFOLD.md: every file written, every command run, anything you could not verify and why, and the result of the project's lint and test gate if you could run it.
