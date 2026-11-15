---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-arch choose a pagination strategy for an activity-feed API. New items are inserted constantly, the feed is sorted newest first by created_at (many items share the same created_at second), and users only scroll forward and never jump to a page number. Save the recommendation to ./pagination.md covering: the chosen strategy, the exact sort and cursor key (including how ties are broken), the maximum page size, and one rejected alternative with the reason it fails for this feed.
