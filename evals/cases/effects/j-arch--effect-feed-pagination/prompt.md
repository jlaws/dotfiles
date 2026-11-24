---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "choose pagination from the architecture references: a cursor with a unique tiebreaker for a feed that takes inserts, and offset rejected because it skips and duplicates rows; .claude/commands/j-arch.md; hard because a created_at-only cursor looks correct until items share a timestamp"
---

/kb:j-arch choose a pagination strategy for an activity-feed API. New items are inserted constantly, the feed is sorted newest first by created_at (many items share the same created_at second), and users only scroll forward and never jump to a page number. Save the recommendation to ./pagination.md covering: the chosen strategy, the exact sort and cursor key (including how ties are broken), the maximum page size, and whether offset/limit pagination would work for this feed, with the reason.
