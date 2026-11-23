---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

I want Claude to automatically run our linter every time it finishes editing a file, and to never touch anything under migrations/. How do I wire that up so it is enforced rather than just asked nicely?
