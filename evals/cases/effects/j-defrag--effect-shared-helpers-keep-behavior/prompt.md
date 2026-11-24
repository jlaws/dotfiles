---
description: "fix clusters in verified batches with one commit each, then report clusters fixed, clusters deferred with the reason, and residual fragmentation; .claude/commands/j-defrag.md; hard because the invalid-email handling is the same job done two ways, which the scan says to converge, yet converging it changes what callers get, so it must be deferred, not fixed"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-defrag shop/. Consolidate the fragmentation you find under shop/ in verified batches (shop/orders.py should end up using the shared name helper in shop/text.py rather than its own copy): the tests must stay green after each batch (`python3 -m unittest`), each duplicated helper is its own cluster with its own commit, and the behavior callers see, including what each function returns or raises for an invalid email, must not change. Save the ranked cluster table (class, files, proposed fix, risk) to ./defrag-plan.md and a closing summary of clusters fixed, clusters deferred with the reason, and residual fragmentation to ./defrag-summary.md. Then run `python3 -m unittest > ./tests.txt 2>&1` and `git log --oneline > ./log.txt`.
