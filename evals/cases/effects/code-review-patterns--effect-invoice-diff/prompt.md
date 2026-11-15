---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Review the diff below for the author and save the review to ./review.md. Requirements: every finding cites file:line from the new file side and carries a priority P0 to P3; tag any finding that no hunk in the diff introduces or exposes as (pre-existing); list at most five nits individually and count the rest as "plus N more nits"; no praise and no suggestions outside this diff; end with a single line `Verdict: PASS`, `Verdict: CONCERNS`, `Verdict: FAIL`, or `Verdict: BLOCKED`. You only have this diff, not the repo.

```diff
--- a/billing/invoice.py
+++ b/billing/invoice.py
@@ -10,9 +10,22 @@ import logging
 
 def charge(card, amount):
     logging.info("charging card %s for %s", card.number, amount)
     return gateway.submit(card, amount)
 
-def apply_discount(total, pct):
-    return total * (1 - pct / 100)
+def apply_discount(total, pct):
+    if pct > 100: pct = 100
+    return round(total * (1 - pct / 100), 2)
+
+def split_payments(total, n, installments=[]):
+    per = round(total / n, 2)
+    for i in range(n):
+        installments.append(per)
+    return installments
+
+def fmt(x): return "$%.2f" % x
+def Total(items): return sum(i.price for i in items)
+def tmp(a, b): return a+b
+def get_stuff(l): return [x for x in l if x]
+def doIt(x): return x*2
+def calc2(q): return q/2
```
