---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "check an experiment split for sample ratio mismatch with a chi-squared test, and when p is below 0.001 halt and never trust the result; .claude/references/workflow/feature-flags-and-ab-testing.md; hard because a 10,240 to 9,760 split looks close to 50/50 and the readout already reports a significant lift"
---

Use the kb:code-reviewer agent to review the change in ./change.diff. Its pull request description is in ./PR.md, including the experiment readout the change relies on, and the files it touches are in the working tree for context. Save the review to ./review.md. Label each finding [blocking], [important], [nit], or [suggestion] and cite the file and line. Say whether the experiment evidence supports shipping the change, and end review.md with a one-line verdict.
