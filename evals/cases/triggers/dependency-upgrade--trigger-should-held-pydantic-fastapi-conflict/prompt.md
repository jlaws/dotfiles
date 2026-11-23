---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

pip fails with ResolutionImpossible between pydantic 2.x and our pinned fastapi 0.85. We need pydantic 2 for a security fix. How do we untangle this?
