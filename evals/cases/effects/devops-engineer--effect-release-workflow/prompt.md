---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "fix a release workflow so a newer run never cancels a deploy, the changelog checkout fetches full history, and the reusable workflow gets explicit secrets instead of inherit; .claude/agents/devops-engineer.md; hard because the shallow default checkout and cancel-in-progress both look fine until a second tag lands"
---

Use the kb:devops-engineer agent to fix the release workflow in ./.github/workflows/release.yml and save the corrected file to ./release.fixed.yml (leave the original unchanged; ./.github/workflows/deploy.yml shows what the called workflow needs). Requirements: if a second tag is pushed while a release is deploying, the running deployment must not be interrupted; the changelog job must list every commit since the previous tag, however long the history is; the deploy job passes the called workflow only the secrets it declares, not every secret in the repository. Also save ./release-notes.md with one short paragraph per change explaining why it was needed.
