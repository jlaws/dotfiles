---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "containerize a Python app with a multi-stage build, a non-root final stage, a cached dependency layer, and an ignore file that excludes git, virtualenvs, caches, and tests; .claude/commands/j-devops.md; hard because USER is easily set only in the build stage and build tools leak into the final image"
---

/kb:j-devops containerize a Python 3.12 FastAPI app. It has requirements.txt, an app/ package, and a tests/ directory; the server starts with uvicorn app.main:app on port 8000. Save the image definition to ./Dockerfile and the ignore file to ./.dockerignore. Requirements: a multi-stage build that keeps build tooling out of the final image; the final image runs as a non-root user; dependency installation is cached separately from application source changes; the ignore file excludes .git, virtualenvs, caches, and tests.
