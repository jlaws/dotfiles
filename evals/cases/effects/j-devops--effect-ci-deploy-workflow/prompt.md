---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-devops write a GitHub Actions workflow to ./ci.yml for a Node project. On pull requests and pushes to main it runs lint and tests. On pushes to main only, after tests pass, it runs a deploy job to a production environment. Requirements: every third-party action is pinned to a full 40-character commit SHA (use an obviously fake placeholder SHA with a comment if you cannot know the real one); permissions are declared per job and each job gets only what it needs; two deploys can never run at the same time, and a deploy that is already running must not be cancelled by a newer push.
