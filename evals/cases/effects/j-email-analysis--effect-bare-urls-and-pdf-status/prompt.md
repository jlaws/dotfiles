---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-email-analysis process the newsletter at inbox/2026-11-01-newsletter.txt and save the report to ./emails/report.md. No network is available, so no PDF can be downloaded: the PDF status cell for each paper must read "failed: " followed by a reason, never "downloaded". Every URL in the report, in prose and in table cells alike, must be a markdown link of the form [text](url), never a bare URL, and a table row must not carry a second link inside its description cell. Every specific paper, repo, tool, or article that the summary or topic overview mentions must carry an inline link to the URL the newsletter cited. End with a Stats section giving total links processed and papers downloaded out of papers found.
