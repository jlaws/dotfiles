# Task: Harvest Agent Memory Into Versioned References

## Context
Found while planning `feat/harness-best-practices`, which removed `memory: user` from the reviewer agents (`docs/adr/workflow/agents-carry-no-persistent-memory.md`).

## Problem
`~/.claude/agent-memory/{security-reviewer,code-reviewer,test-writer}` hold review lessons that no longer load into any agent. The durable ones are lost unless they move into versioned guidance.

## Scope
1. Read each store and keep only lessons that generalize beyond one project.
2. Fold them into `references/security/`, `references/testing/`, and `code-review-patterns`, in both the `.claude/` and `.agents/` trees, keeping reference section parity.
3. Scrub project names, repo names, and paths.
4. Leave the stores in place; deleting them is the user's call.
5. Run `make verify`.
