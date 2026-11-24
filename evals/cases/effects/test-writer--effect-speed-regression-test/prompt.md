---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "a timing test discards warm-up iterations, takes more than 30 samples, compares the median or a high percentile rather than the mean, and allows 5 to 10 percent of noise before failing; .claude/references/testing/performance-testing-and-profiling.md; hard because the common habit is the minimum or mean of a few timeit loops with a hand-picked margin"
---

Use the kb:test-writer agent to add a speed regression test for slugify in textkit/slug.py. slugify replaced an older implementation that is kept as _slugify_reference, and the test must catch slugify becoming slower than _slugify_reference. Put it in tests/test_slug_perf.py using unittest and time.perf_counter, with no third-party packages. Choose how many timing samples to take, what to do before timing starts, which statistic of the samples to compare, and how much slower slugify may measure before the test fails, and give the reasons in a comment at the top of the test file. Leave textkit/slug.py as it is at the end. Run `python3 -m unittest tests.test_slug_perf -v` and save its full output to ./perf-output.txt.
