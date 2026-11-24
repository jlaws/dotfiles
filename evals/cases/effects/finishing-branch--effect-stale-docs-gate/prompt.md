---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "validate docs before pushing so the pushed branch carries the doc fix, then open a PR with a real body; .claude/skills/finishing-branch/SKILL.md; hard because the branch commit claims the rename but leaves README stale, and a model can fix the README after the push so the reviewed branch stays stale"
---

The work on feat/rename-output is done: the command line flag is now --output. Get it to the point where a reviewer can look at it; README.md must document the flag as it now is. Afterward save `git --git-dir=.remote.git for-each-ref --format='%(objectname)%09%(refname)' refs/heads > ./remote-heads.txt` and `git --git-dir=.remote.git show feat/rename-output:README.md > ./pushed-readme.md`. There is no network, so if the pull request cannot be created from here, still make the attempt with the full title and body. The body must have a summary, what verification you ran and its result, and how risky the merge is and how to revert it.
