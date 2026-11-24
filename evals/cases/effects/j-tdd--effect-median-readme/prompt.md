---
description: "write tests first, confirm they fail for the missing behavior and not an import error, implement, verify the full suite, update docs, and commit; .claude/commands/j-tdd.md; hard because a test file importing a function that does not exist yet fails with ImportError, which a model takes as the red run"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-tdd add a median(xs) function to calc/ops.py. For an even number of items it returns the average of the two middle values; for an odd number, the middle value. It raises ValueError on an empty list and must not reorder or otherwise mutate the list passed in. Write the tests first in tests/test_median.py, using literal expected values. Run them before implementing and save that run to ./red-run.txt: it must show failing assertions from the missing behavior, not an ImportError, so add a stub first if you need to. Then implement, run the full suite with `python3 -m unittest`, document median in README.md, and commit.
