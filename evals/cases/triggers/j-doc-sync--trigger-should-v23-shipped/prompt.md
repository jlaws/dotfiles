---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "a user shipped a release and asks to fix the README and docs that are now stale; .claude/commands/j-doc-sync.md; should"
---

We shipped v2.3 yesterday: a new `--dry-run` flag on the CLI, the `LEGACY_MODE` env var is gone, and there is a new `/v2/exports` endpoint. Go through the README and everything under docs/ and fix whatever is now stale.
