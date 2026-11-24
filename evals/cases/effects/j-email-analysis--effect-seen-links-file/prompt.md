---
description: "with no seen-links header in the email, fall back to the .seen-links file in the working directory and skip every URL it lists; .claude/commands/j-email-analysis.md; hard because the record of processed links is a dotfile the email never mentions, so without the header a model concludes nothing was seen and reports every link"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-email-analysis process the newsletter at inbox/2026-11-15-newsletter.txt and save the report to ./emails/report.md. I run this every week, so skip every link an earlier run already processed. No network is available, so nothing can be downloaded or resolved. End with a Stats section that counts the links skipped as already seen.
