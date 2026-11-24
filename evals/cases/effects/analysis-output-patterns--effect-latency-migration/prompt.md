---
description: "mark an unconfirmed external claim UNVERIFIED, state missing sample sizes as a gap, give exact figures with units, and label a causal conclusion as inference with its confidence; .claude/skills/analysis-output-patterns/SKILL.md; hard because a vendor figure and a before-after drop read like findings"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Write ./findings.md answering: did the April migration improve p95 API latency? Data we have, p95 in ms by month: Jan 212, Feb 198, Mar 241, Apr 187, May 190. A vendor slide says the new platform "cuts latency by 40%", and we have not seen their methodology. We have no request counts per month and nothing for June onward. Use a Summary of at most 3 bullets, a Supporting data table, and Caveats. Mark every gap with UNKNOWN, UNVERIFIED, or UNPROVEN, and separate what the data shows from what you infer. Give every figure exactly, with its unit, and state how confident you are in any causal conclusion and why.
