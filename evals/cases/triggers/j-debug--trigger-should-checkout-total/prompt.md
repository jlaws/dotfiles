---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Our test_checkout_total started failing after yesterday's merge:

```
FAIL: test_checkout_total (tests.test_cart.CartTests)
AssertionError: 99.9 != 100.0
```

Nothing in the test changed and I cannot see why the total is off by 0.1. Can you track down what is actually causing it?
