---
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "adding inline comments to one function; .claude/commands/j-docs.md; near: none"
---

Add short inline comments to this function explaining each step:

```python
def dedupe(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
```
