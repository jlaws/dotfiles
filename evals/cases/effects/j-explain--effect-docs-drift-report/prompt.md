---
description: "a findings-first information report, every statement labeled data or inference, self-contained, with an export button; .claude/commands/j-explain.md; hard because a report habitually opens with background on the tool, and the evidence-or-inference split is easy to blur when the rejection is read from code"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-explain produce an information report on whether this tool's README matches its code. Save it to ./drift-report.html as one self-contained file with inline CSS and JS and no external resources. Lead with the finding before any background. Label every statement as either evidence (read directly from a file, with the file named) or inference (your reasoning about what users will hit). Color-code findings by severity, and include a button that copies or downloads the findings.
