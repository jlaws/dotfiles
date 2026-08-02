---
name: subagent-driven-development
description: "Use when executing a plan with isolated tasks delegated."
allowed-tools: Task, Read, Grep, Glob, Bash
skills:
  - executing-plans
---

# Subagent-Driven Development

**Core principle:** Fresh subagent per task + per-task review (spec + quality) + broad final review = high quality, fast iteration.

Each task runs in an isolated subagent context instead of inheriting session history. The orchestrator (you) stays thin: dispatch, review, integrate. This keeps the main context small enough to run for hours without compaction.

## When to Use (vs `executing-plans`)

| Signal | Use `subagent-driven-development` | Use `executing-plans` (inline) |
|--------|-----------------------------------|-------------------------------|
| Plan size | Large (many tasks, long) | Small / medium |
| Task coupling | Mostly independent | Tightly coupled, shared state |
| Context pressure | High — inline would compact | Low |
| Task specs | Complete, self-contained | Need live back-and-forth |

Both modes share the same living-document ledger (below). This skill is the *dispatch* mode; `executing-plans` is the *inline batch* mode.

## The Process

### Step 0: Plan Pre-Scan

Read the whole plan once before Task 1. Flag internal contradictions, inconsistent signatures across tasks, or ordering problems. Raise them with the user before dispatching anything.

### Step 1: Per-Task Cycle

For each task in order:

1. **Extract the task brief.** Write the single task's requirements (behavioral check, files, steps, acceptance) to a scratch file, e.g. `scratchpad/briefs/task-NN.md`. Do NOT paste the whole plan into the dispatch prompt.
2. **Dispatch the implementer subagent** (via the Task tool) with: the brief file path, the model for this task (see Model Selection), explicit constraints (what NOT to touch), and the deliverable (code + passing tests + one commit). Answer clarifying questions if the implementer asks.
3. **Implementer self-reviews and commits.** It follows TDD (failing test -> implement -> pass), commits, and reports the commit range.
4. **Generate the review package.** `git diff <task-base>..<task-head>` — capture to a scratch file rather than pasting a large diff into the reviewer prompt.
5. **Dispatch the task reviewer** (via the Task tool). It MUST return **both** verdicts, **spec compliance first, then code quality**:
   - **Spec compliance** — does the diff satisfy the task brief's behavioral check and acceptance? Flag any acceptance item that **cannot be confirmed from the diff alone** for the orchestrator to resolve — never assume it holds. Treat any **unrequested addition** (extra code, deps, files beyond the brief) as a finding: scope creep is a failure, not a bonus.
   - **Code quality** — smells, edge cases, error handling, naming, test quality.
   Reviewer output is a severity-labeled findings list (Critical / Important / Nit).
6. **Fix loop.** If Critical/Important findings exist, dispatch a fix subagent with the findings file, then re-review. Repeat until clean. Never dismiss findings pre-emptively in the reviewer prompt.
7. **Mark the task done** in the ledger with its commit range.

### Step 2: Final Whole-Branch Review

After all tasks: dispatch one reviewer on the **most capable model** over the entire branch diff (`git diff main...HEAD`) for cross-cutting concerns no single-task review can see — architecture drift, integration gaps, inconsistent patterns, and stale documentation (apply `documentation-validation`). Resolve findings, then hand off to `finishing-branch`, which opens the PR without being asked.

## File-Based Handoffs

Everything you paste into a dispatch prompt stays resident in your context. Hand artifacts over **as files**:

- **Task brief** -> `scratchpad/briefs/task-NN.md` (extracted from the plan).
- **Review package** -> `git diff <base>..<head>` captured to a scratch file; pass the path.
- **Findings** -> reviewer writes findings to a file; fix subagent reads it.

No helper scripts required — `git diff` + a scratch file is the whole mechanism.

`scratchpad/` is gitignored working space and must never be committed. If the repo does not ignore it yet, add it to `.gitignore` before writing the first brief.

## Model Selection

Always specify the model explicitly when dispatching (subagents otherwise inherit an expensive default).

| Task complexity | Model tier |
|-----------------|-----------|
| Mechanical, 1-2 files, complete spec | cheap/fast |
| Multi-file integration | standard |
| Architecture, ambiguous, final review | most capable |

## Progress Ledger

Maintain the same living-document sections as `executing-plans`, in the plan file, so work survives compaction:

```markdown
## Progress
- [x] Task 1: Setup schema — `a1b2c3d..e4f5g6h`
- [ ] Task 2: Model layer ← current

## Decision Log
| Task | Decision | Rationale |
|------|----------|-----------|

## Surprises & Discoveries
- Task 1: ...
```

## Red Flags

- Pasting the full plan or a large diff into a dispatch prompt (use files)
- Dispatching without specifying the model
- Skipping the reviewer, or only getting one of the two verdicts
- Softening the reviewer prompt to avoid findings
- Marking a task done with Critical/Important findings unresolved
- Letting the orchestrator start editing task code itself (it dispatches, it does not implement)

## Integration

**Receives plans from:** `writing-plans`
**Alternative to:** `executing-plans` (inline mode) — same ledger, different dispatch
**Uses:** `verification-before-completion` for every verdict; `dispatching-parallel-agents` when independent tasks can run concurrently
**Hands off to:** `finishing-branch` after the final review — which opens the PR, it does not ask whether to
