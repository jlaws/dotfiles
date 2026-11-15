---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-email-analysis process the newsletter at inbox/2026-11-08-newsletter.txt and save the report to ./emails/report.md. The block at the top of the file lists links already processed: none of those links may appear anywhere in the report. Leave out the unsubscribe and view-in-browser links entirely. Keep the click-tracking redirect link for the new product verbatim as a markdown link, and do not replace it with placeholder text. No network is available, so nothing can be downloaded or resolved. End with a Stats section that counts the links skipped as already seen.
