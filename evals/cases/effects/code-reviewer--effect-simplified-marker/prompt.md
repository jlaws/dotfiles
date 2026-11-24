---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "leave a SIMPLIFIED shortcut with a ceiling and upgrade path unflagged, but flag a SIMPLIFIED marker on a security control as blocking and name the marker itself as the finding; .claude/agents/code-reviewer.md; hard because a reviewer tends to either trust every deliberate-shortcut comment or flag every shortcut alike"
---

Use the kb:code-reviewer agent to review ./app/store.py and ./app/auth.py, which are new code. Save the review to ./review.md. Label each finding [blocking], [important], [nit], or [suggestion] and cite file:line. Some shortcuts in the code are commented as deliberate, with a stated size limit and upgrade path. Leave a deliberate shortcut alone when it is a plain simplification, but flag anything where a shortcut weakens security regardless of the comment, and for those say that the comment itself is the problem.
