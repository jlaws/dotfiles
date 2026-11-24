---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "restructuring a slow GitHub Actions pipeline and serializing deploys; .claude/commands/j-devops.md; should"
---

Our GitHub Actions pipeline takes 25 minutes: install, lint, unit tests, Docker build, then deploy. Help me restructure it so it is faster and the deploy cannot run twice at once.
