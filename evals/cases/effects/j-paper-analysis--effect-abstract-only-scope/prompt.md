---
tags: [effects, effect, heavy]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-paper-analysis analyze the paper whose abstract is in ./abstract.txt and save the analysis to ./analysis.md. I only have the abstract; the full text is paywalled and I have not seen it. Anything that needs the full text, such as methods, experimental setup, baselines, ablations, and reproducibility, you cannot assess: mark it "not assessed" instead of guessing at weaknesses. Still write the header with a one-sentence TL;DR that states the headline result from the abstract with its numbers, the five Cs as far as the abstract supports them, and a reviewer assessment whose confidence rating (1 to 5) reflects that you read only the abstract. Do not look anything up online and do not write any URLs.
