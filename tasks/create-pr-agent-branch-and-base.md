# Task: Align the create-pr Agent's Branch Name and Diff Base

## Context
Found while planning `feat/harness-best-practices`.

## Problem
`.claude/agents/create-pr.md:21` derives the branch as `<username>/<short-kebab-description>`, which contradicts the `type/short-description` convention in the always-loaded Git guidance. `.claude/agents/create-pr.md:37` builds the PR body from `git log --oneline main..<branch>`, a local `main` that may be stale against `origin/main`. The same text lives in `.codex/agents/create-pr.toml` and `.gemini/config/agents/create-pr.md`.

## Scope
1. Add a failing test in `tests/test_agent_config.py` that the create-pr agent in each tree names `type/short-description` and diffs against `origin/main`.
2. Fix the branch rule and the diff base in all three agent trees.
3. Run `make verify`.
