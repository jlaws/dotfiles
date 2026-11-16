---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:research-analyst agent to review this results claim and say whether it holds. Use only the material in this prompt; do not search the web. "Our new method beats all four baselines. We trained each system with 3 random seeds on the same data splits and measured 5 metrics. After seeing the results we picked F1, where our mean was highest, and ran a t-test against each baseline; all four gave p < 0.05, so the method is significantly better. The largest gain is 0.4 F1 points." Save the review to ./stats-review.md. Name each methodological flaw, say why it matters, and give a concrete fix for it, including the specific correction method for the four baseline comparisons and a seed count with how to justify it. Finish with a verdict on whether the claim is supported.
