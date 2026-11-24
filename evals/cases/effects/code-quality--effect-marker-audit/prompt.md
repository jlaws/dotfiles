---
description: "a SIMPLIFIED marker needs a ceiling and an upgrade path, is never valid on a security control, a sanctioned one is not a finding, and a bare TODO is a finding; .claude/skills/code-quality/SKILL.md; hard because a marker on the login shortcut reads as a sanctioned decision unless one knows the security exclusion"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Review the file below (src/store.py) for quality problems and save your findings to ./quality-review.md, one line per finding formatted `file:line - severity - problem`, followed by one line `Sanctioned: ` naming any shortcut that you judge is a legitimate, acceptable deliberate simplification and therefore not a finding.

```python
 1  import threading
 2
 3  USERS = {"ana": "hunter2", "raj": "correct-horse"}
 4  ITEMS = []
 5  LOCK = threading.Lock()
 6
 7  # SIMPLIFIED: plain-text password compare, hash when real accounts exist
 8  def login(user, pw):
 9      return USERS.get(user) == pw
10
11  # SIMPLIFIED: linear scan, breaks past ~50k items (p95 over 100 ms), switch to a dict keyed by sku
12  def find_item(sku):
13      for item in ITEMS:
14          if item["sku"] == sku:
15              return item
16      return None
17
18  # SIMPLIFIED: single global lock
19  def reserve(sku, qty):
20      with LOCK:
21          item = find_item(sku)
22          item["stock"] -= qty
23          return item
24
25  # TODO
26  def export_csv(path):
27      pass
```
