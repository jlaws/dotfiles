---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "for a dbt incremental model of late-arriving date-partitioned events, re-read at least 3 days of data on every run (three daily partitions or a 3-day interval) and keep re-read rows from duplicating; .claude/references/data/analytics-and-transformations.md; hard because the events arrive at most 24 hours late, so a 1-day lookback looks sufficient"
---

/kb:j-data make the dbt model ./models/marts/fct_page_views.sql incremental; the facts about the data are in ./docs/events.md. Choose the incremental strategy, how far back each incremental run re-reads, and how re-read rows avoid duplicates. Edit the model in place and save your reasons to ./incremental-notes.md.
