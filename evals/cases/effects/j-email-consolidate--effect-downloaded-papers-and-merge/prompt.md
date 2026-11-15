---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-email-consolidate merge the per-email reports in ./emails into one weekly digest at ./digest.md. Skip any file whose name ends in -summary.md. Requirements: the same repo, tool, or article described in several reports appears once. A paper whose row in a report's Research Papers table has PDF status "downloaded" is already in my library, so leave that table row out; a paper that is only mentioned in prose in some other report keeps its link, and a paper whose PDF status is a failure stays under Research Papers with "(needs manual download: <reason>)" in its note. Drop sponsor placements. Do not link back to the per-email files. The Summary section has two paragraphs: the week's headlines first, then the cross-paper themes. When you finish, no leftover HTML comment markers may remain in digest.md.
