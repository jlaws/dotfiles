---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Review the file below (src/store.py) for quality problems and save your findings to ./quality-review.md, one line per finding formatted `file:line - severity - problem`, followed by one line `Sanctioned: ` naming any shortcut that you judge is a legitimate, acceptable deliberate simplification and therefore not a finding.

```python
 1  USERS = {"ana": "hunter2", "raj": "correct-horse"}
 2  ITEMS = []
 3
 4  # SIMPLIFIED: plain-text password compare, hash when real accounts exist
 5  def login(user, pw):
 6      return USERS.get(user) == pw
 7
 8  # SIMPLIFIED: linear scan, breaks past ~50k items (p95 over 100 ms), switch to a dict keyed by sku
 9  def find_item(sku):
10      for item in ITEMS:
11          if item["sku"] == sku:
12              return item
13      return None
14
15  # SIMPLIFIED: single global lock
16  def reserve(sku, qty):
17      item = find_item(sku)
18      item["stock"] -= qty
19      return item
20
21  # TODO
22  def export_csv(path):
23      pass
```
