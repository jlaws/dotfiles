---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-rebase Add summarize helper. The pull request for feat/summarize (#31) is open and I'm on that branch with everything committed. gh isn't available here, so treat the open-PR check as satisfied and skip the PR text update, but do the rest. Bring the branch up to date with main as a single commit using the sentence above as its message, and push it to origin. The new helper must still work against main's current code. Then run `python3 -m unittest > ./tests.txt 2>&1`, `python3 -c "from calc.summary import summarize; print(summarize([1, 2, 3]))" > ./run.txt 2>&1`, `git rev-list --left-right --count origin/main...HEAD > ./counts.txt`, and `git log --format=%s origin/main..origin/feat/summarize > ./pushed.txt`. Write ./report.md naming every file on the branch that depended on something main changed (whether or not you edited it) and saying which of those files has no test that covers it.
