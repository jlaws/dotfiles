---
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Our GitHub Actions workflow takes 25 minutes because it reinstalls dependencies and runs every test on every push. Help me speed up the pipeline config with caching and a job matrix.
