---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:code-reviewer agent to review the change in ./change.diff (a patch titled "Make CI green again" for a small Python service; the files it touches are in the working tree for context). Save the review to ./review.md. Give every finding a severity label in square brackets: [blocking], [important], [nit], or [suggestion], and cite the file and line for each. I care most about whether the patch fixes the failures or only makes the checks stop failing, so say so for each change that lowers a bar. End review.md with a one-line verdict.
