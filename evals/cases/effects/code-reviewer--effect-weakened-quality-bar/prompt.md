---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "flag a patch that turns CI green by lowering a coverage threshold, skipping a failing test, or silencing a checker as a weakened quality bar, not a fix; .claude/agents/code-reviewer.md; hard because each change is small and the skip is labelled flaky"
---

Use the kb:code-reviewer agent to review the change in ./change.diff (a patch titled "Make CI green again" for a small Python service; the files it touches are in the working tree for context). Save the review to ./review.md. Give every finding a severity label in square brackets: [blocking], [important], [nit], or [suggestion], and cite the file and line for each. I care most about whether the patch fixes the failures or only makes the checks stop failing, so say so for each change that lowers a bar. End review.md with a one-line verdict.
