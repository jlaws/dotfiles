---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

The work on feat/rename-output is done: the command line flag is now --output. Get it to the point where a reviewer can look at it; README.md must document the flag as it now is. Afterward save `git ls-remote --heads origin > ./remote-heads.txt`. There is no network, so if the pull request cannot be created from here, still make the attempt with the full title and body. The body must have a summary, what verification you ran and its result, and how risky the merge is and how to revert it.
