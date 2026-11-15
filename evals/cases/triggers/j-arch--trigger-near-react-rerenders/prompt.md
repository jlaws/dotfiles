---
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Our React search page re-renders the whole results list on every keystroke in the filter box. How do I stop that? Component code is a standard function component with useState for the query.
