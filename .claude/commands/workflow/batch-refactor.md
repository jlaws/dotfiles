---
name: batch-refactor
description: "Parallel batch refactoring across many files using worktree-isolated agents. Use when refactoring a concept, pattern, or API across 10+ files. Do NOT use for small refactors touching <5 files (just do it directly) or behavior changes (use /brainstorm + /write-plan)."
argument-hint: "<refactoring description>"
---

Load and follow these skills:
- `workflow/multi-agent-development` for agent coordination and worktree isolation
- `workflow/using-git-worktrees` for worktree setup
- `workflow/refactoring-and-debt` for safe refactoring discipline (TEST → REFACTOR → VERIFY → COMMIT)

Refactoring: $ARGUMENTS

If no arguments provided, ask what the user wants to refactor.

## Coordination Plan

Follow this sequence:

### Phase 1: Discovery
1. Identify EVERY file that needs changes (use Grep/Glob exhaustively)
2. Group files into independent batches (default: 5 batches). Files in the same batch must not depend on each other.
3. Present the batch plan to the user for approval before proceeding.

### Phase 2: Parallel Execution
For each batch, spawn a Task agent with `isolation: "worktree"`:

```
Agent(
  prompt="<batch-specific instructions with file list and refactoring rules>",
  isolation="worktree",
  subagent_type="general-purpose"
)
```

Each agent must:
- Make the specified changes to its assigned files only
- Run lint/format on changed files
- Commit changes to its isolated branch
- NOT clean up its worktree (parent handles merge)
- NOT invoke finishing-branch skill

Pre-flight: verify each worktree is fully isolated before dispatching.

### Phase 3: Integration
After all agents complete:
1. Create a single PR branch from main
2. Sequentially merge each agent's branch, resolving conflicts
3. Run full lint validation on the merged result
4. Run tests if applicable

### Phase 4: PR
Use `workflow/create-pr` skill to open the PR with a summary of all changes.
