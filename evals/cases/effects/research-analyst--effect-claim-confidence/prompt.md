---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:research-analyst agent to assess how reliable these four claims are from the sources I collected. Claim A: "Library X is 3x faster than Y". Sources: one blog post by the author of X. Claim B: "Drug D lowers outcome O". Sources: a peer-reviewed trial, a second peer-reviewed trial, and a regulator's published review, all agreeing. Claim C: "Tool T supports feature F". Sources: T's official docs say yes; a forum post from last year says no; the changelog says F was added after that post. Claim D: "Method M generalizes to domain Z". Sources: two community blog posts that agree with each other. Save the assessment to ./claims.md with one row per claim giving a confidence from 0.0 to 1.0, the sources and their quality, how many independent sources agree, and any contradiction. A claim backed by a single source must not score above 0.6, and a contradicted claim must be flagged with both positions stated and scored 0.2 lower than it would otherwise be.
