---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "a behavior change is not done until every doc that describes the old behavior, including reference docs and the changelog, describes the new one; .claude/skills/documentation-validation/SKILL.md; hard because the old default also sits in docs/cli.md, away from the README a model checks first, and a model touching the changelog tends to rewrite the released 0.1.0 entry that must stay as history"
---

Change tool.py so that when --output is not given, it writes the JSON lines to a file next to the input, named after the input with a .jsonl extension (data.csv becomes data.jsonl), instead of printing to stdout. Update the help text and docstring to match.

The repo's documents are README.md, docs/cli.md, and CHANGELOG.md. Finish the change the way you would before committing it: every document in this repo that describes the old default must describe the new one, and the changelog must record the change for users without rewriting what earlier releases did. Then write ./change-notes.md with one line starting `Docs:` that says which docs you updated or N/A with a reason.
