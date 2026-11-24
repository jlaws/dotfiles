---
description: "detect bibliography duplicates by normalized title, flag entries missing year or venue, and propose merges without deleting entries; .claude/commands/j-paper.md; hard because the duplicate pair has different keys and title casing, so a key-based check misses it"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-paper check paper/refs.bib for duplicate entries and incomplete entries. Write your findings to ./bib-report.md. Detect duplicates by normalized title (case and punctuation ignored), not just by key, and list each duplicate pair with both keys and which one you propose keeping. Flag entries missing a DOI, year, or venue, naming the missing fields per key. Propose merges only: do not edit or delete anything in refs.bib.
