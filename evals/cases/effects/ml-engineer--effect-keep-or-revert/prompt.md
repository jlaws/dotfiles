---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "keep a change only when train and test both rise beyond noise, revert a train-only gain, and adopt a cost-at-parity change only when the largest loss its interval allows sits inside the margin fixed before the run; .claude/references/ai-ml/eval-and-benchmarking.md; hard because a train gain with a flat test looks like progress, and a cost change whose point estimate is inside the margin looks safe while its interval is not"
---

Use the kb:ml-engineer agent to decide, for each of five changes we tried on the system prompt of our support-bot LLM, whether to keep it or revert it, and why. Scores are task pass rates in points on two sets: train (the tasks we tune against) and test (held-out tasks the tuner never sees). Each delta is against the current baseline, with a 95 percent paired bootstrap interval over tasks. Changes A, B, and C aim to raise quality. Changes D and E aim to cut cost at equal quality: D trims the system prompt (30 percent fewer input tokens) and E folds two tool calls into one (20 percent lower cost per task). Before running D and E we fixed a margin: a cost change may lose at most 2 points on test.

| Change | Train delta [95% interval] | Test delta [95% interval] |
|---|---|---|
| A | +4.0 [+1.5, +6.5] | +3.0 [+0.8, +5.2] |
| B | +5.0 [+2.0, +8.0] | +0.5 [-2.0, +3.0] |
| C | +1.0 [-1.5, +3.5] | +1.2 [-1.0, +3.4] |
| D | -0.5 [-1.8, +0.8] | -0.8 [-2.6, +1.0] |
| E | -0.2 [-1.2, +0.8] | -0.6 [-1.5, +0.4] |

Save the decisions to ./keep-or-revert.md, one line per change giving keep or revert and the reason.
