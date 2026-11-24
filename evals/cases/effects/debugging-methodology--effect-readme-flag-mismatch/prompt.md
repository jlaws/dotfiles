---
description: "reproduce first, classify the failure locus, and fix the fault where it lives, here the documentation; .claude/skills/debugging-methodology/SKILL.md; hard because adding an --out alias to the code is the quickest way to make the command work"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

A user reports that the README example for this tool fails: `python3 tool.py --input data.csv --out data.jsonl` exits with "unrecognized arguments: --out". Find the root cause and fix it where the fault actually lives.

Reproduce the failure first by running the command against a small CSV you create, then fix, then re-run to confirm. Save ./investigation.md with: the root cause with file:line, which part is at fault (the tool's code, its documentation, its tests, or the environment) and why, the exact repro command with its observed output, and the fix with the verification run.
