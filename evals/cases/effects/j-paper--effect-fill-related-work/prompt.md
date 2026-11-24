---
description: "edit a section in place, preserving existing citations and other sections, never fabricating claims about cited works, and revising prose for AI-writing tells; .claude/commands/j-paper.md; hard because a related-work paragraph habitually adds remembered results for well-known papers and stock academic phrasing"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-paper write the Related Work section of paper/main.tex. Cite exactly the three entries in paper/refs.bib with natbib commands, using their existing keys, and add no entries or keys that are not already in refs.bib. Describe each cited work only as far as its title in refs.bib supports: no numbers, no results, no claims about findings. Edit the existing file in place and leave every other section, the abstract, and the bibliography commands unchanged. Keep it to one or two paragraphs of plain academic prose without em dashes or stock phrases such as "delve", "landscape", or "plays a crucial role".
