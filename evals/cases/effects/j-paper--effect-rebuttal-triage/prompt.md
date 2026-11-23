---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-paper draft a rebuttal for the reviews in ./reviews.md of the paper in ./paper/main.tex. Write a triage table covering every reviewer comment, keyed by the comment IDs in reviews.md, to ./triage.md, then the rebuttal to ./rebuttal.md. The venue allows 400 words and no external links. Do not claim any change that is not already in main.tex; where a reply needs new work, say what the authors would have to confirm. Where main.tex already answers a comment, point to where. Correct factual errors in the reviews.
