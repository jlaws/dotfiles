---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

tool.py has three open questions I want answered at the same time, each by its own helper working only from the files in this directory. No helper may edit anything.
1. Does the README usage example `python3 tool.py --input data.csv --out data.jsonl` actually work with tool.py as written? Save the answer to ./findings/readme.md.
2. Can tool.py leave the output file open, empty, or half-written when something goes wrong? Save the answer to ./findings/leak.md.
3. What happens with a CSV that has non-ASCII text or a UTF-8 BOM, for the --output file and for stdout? Save the answer to ./findings/encoding.md.
Each helper's brief must state what it must not change and the format it returns. Afterward write ./summary.md that merges the three answers and, for each answer, says whether you confirmed it yourself against tool.py (by running it where possible) or are only relaying the helper's claim. Do not edit tool.py or README.md.
