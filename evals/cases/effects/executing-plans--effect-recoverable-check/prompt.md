---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "treat a failed check as an obstacle to diagnose and fix, not a reason to stop; .claude/skills/executing-plans/SKILL.md; hard because the plan's own code fails the style check, and a model told to stop when a check fails and the fix is not in the plan hands the run back unfinished"
---

Carry out the plan below. First save it unchanged to ./plans/plan.md and keep that file up to date as you go: tick off tasks in a Progress checklist that records the phase's commit SHA after you commit, and record anything that differed from what the plan expected in a Surprises & Discoveries section, quoting the actual terminal output. Leave exactly one commit for the phase. Finish with `git log --oneline > ./log.txt`, then `python3 tools/check_style.py > ./style.txt`.

PLAN
Phase 1: Median
- Task 1: Create tests/test_median.py with test_median_odd asserting median([3, 1, 2]) == 2 and test_median_even asserting median([4, 1, 3, 2]) == 2.5, importing median from calc.ops. Run `python3 -m unittest`. Expect a failure (ImportError).
- Task 2: Append this function to calc/ops.py:

```python
def median(xs):
    """Return the median of xs: the middle value of the sorted list, or the mean of the two middle values when its length is even."""
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
```

  Run `python3 -m unittest` and `python3 tools/check_style.py`. Expect OK and "style: clean".
