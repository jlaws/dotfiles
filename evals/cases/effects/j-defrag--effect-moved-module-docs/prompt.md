---
description: "when a cluster moves public surface such as an import path, update the docs that name it before that cluster's commit; .claude/commands/j-defrag.md; hard because the merge is the obvious edit and every importer the tests catch is a Python file, while the old import paths also sit in README examples that no test runs"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-defrag calc/. The formatting helpers are over-split: calc/fmt_money.py and calc/fmt_percent.py each hold one function. Merge them into one module, calc/fmt.py, and delete the two old modules; nothing outside this repository imports them, so add no compatibility shims. Work in verified batches with the tests green after each (`python3 -m unittest`), one commit per cluster, and make each commit one you would ship as is. Then run `python3 -m unittest > ./tests.txt 2>&1` and `git log --name-status --format='commit %h %s' > ./log.txt`.
