---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

I am setting up agent instructions for a monorepo. Some rules apply to every project I work on, some to this repo, and some to just the api package. Where should each kind of rule live, and how do I keep the always-loaded part from bloating?
