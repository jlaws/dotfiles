---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Write ./findings.md answering: did the April migration improve p95 API latency? Data we have, p95 in ms by month: Jan 212, Feb 198, Mar 241, Apr 187, May 190. A vendor slide says the new platform "cuts latency by 40%", and we have not seen their methodology. We have no request counts per month and nothing for June onward. Use a Summary of at most 3 bullets, a Supporting data table, and Caveats. Mark every gap with UNKNOWN, UNVERIFIED, or UNPROVEN, and separate what the data shows from what you infer.
