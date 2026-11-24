---
description: "one self-contained HTML file with inline CSS, JS, and SVG, a spatial diagram of the structure, and a table of contents with jump links; .claude/commands/j-explain.md; hard because the default HTML habit pulls in a CDN font, script, or image, and the docstring-versus-code mismatch on empty input is easy to restate as fact"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-explain explain how the calc package in this repo works, for a new teammate. Save it to ./calc-explainer.html as one self-contained file: inline CSS, inline JS if any, no external resources of any kind. Include an inline SVG diagram of the package structure and function relationships, and a table of contents with jump links to each section. Wherever the code disagrees with its own documentation, call it out visibly with the exact function name, what the docstring promises, and what the code does.
