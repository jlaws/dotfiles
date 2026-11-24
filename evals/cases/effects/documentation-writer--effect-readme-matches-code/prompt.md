---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "correct README flag drift against the code, run the example to prove it, state the behavior in plain prose without stock phrases, and record the fix under Fixed in an Unreleased changelog; .claude/agents/documentation-writer.md; hard because the stale flag looks plausible and generated prose drifts into stock phrases"
---

Use the kb:documentation-writer agent to fix the README in this directory so it matches tool.py. Every flag the README mentions must be one tool.py actually accepts, and the usage example must run: create a small ./data.csv with a header and two rows, run the README's example command against it, then print the file it wrote (for example with cat; skip this if the example prints to stdout), and save the commands and everything they printed, including the converted rows, to ./check.txt. Add a short "Behavior" paragraph (three to five sentences of plain prose, not a list) to README.md that states, from the code, how rows are converted, what types the values have, and what happens to an existing output file. Add a CHANGELOG.md with an Unreleased section that records the README correction under Fixed. Keep the prose plain: no promotional or stock phrases such as seamless, robust, leverage, or "it is important to note".
