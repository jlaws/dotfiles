---
tags: [effects, effect, heavy]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-paper-analysis analyze the paper at ./paper.md and save the analysis to ./analysis.md. It is a local file, so do not look anything up online; for the official code and any other resource you cannot confirm from the file, write "Not available" and invent no links. Cover the five Cs, key results, a critical analysis, and a reviewer assessment with an overall score from 1 to 10, a confidence from 1 to 5, and a recommendation. Calibrate honestly: 6 is marginally above the acceptance threshold and 8 or higher means genuinely strong. In the critical analysis, check whether the reported numbers and error bars look plausible, whether the ablation removes the mechanism the paper claims, and whether the baselines were treated fairly.
