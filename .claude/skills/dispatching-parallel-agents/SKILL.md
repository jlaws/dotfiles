---
name: dispatching-parallel-agents
description: "Use when independent tasks can run concurrently."
allowed-tools: Task, Read, Grep, Glob, Bash
---

# Dispatching Parallel Agents

**Core principle:** One agent per independent problem domain, dispatched concurrently — parallel investigation beats sequential when the problems don't share state.

## When It Works

- Multiple failures with **unrelated** root causes
- Each investigation needs no shared context
- Problems live in **different subsystems**
- Agents won't edit overlapping code

## When to Avoid

- Failures are interconnected (one cause, many symptoms)
- Understanding requires seeing the whole system at once
- Agents would edit the same files (write conflicts)
- The result of one investigation changes how you'd scope the next (sequence instead)

## The Dispatch Mechanism

**Multiple subagent (Task tool) calls in a single message run in parallel.** Sequential messages do not.

Each brief must be:

- **Focused** — one test file, one subsystem, one problem
- **Self-contained** — all background the agent needs; it can't see your context
- **Explicit about constraints** — what it must NOT change
- **Specific about the deliverable** — exact output format (findings file, patch, summary)

Hand large inputs over as files (see `subagent-driven-development` handoffs), not inline text.

## Post-Run Integration

Parallel agents finishing is not the end. After they return:

1. **Check for conflicting edits** — did two agents touch the same file? Reconcile.
2. **Review each summary against the diff** — treat a subagent's report as peer input, not proof. Its summary describes what it meant to do; the diff shows what it did.
3. **Run the full suite** — parallel fixes that each pass in isolation can still break in combination.

## Red Flags

- Dispatching interconnected problems in parallel (findings contradict, fixes fight)
- Briefs that assume the agent shares your context
- No stated constraint, so agents wander outside scope
- Trusting parallel summaries without checking the combined diff + full suite
- Parallelizing work that edits overlapping files

## Orchestration Limits & Patterns

For large or automated fan-outs (e.g. the harness Workflow tool), apply these:

- **Concurrency cap** — run about 16 agents at once (or fewer); excess should queue, not all launch together.
- **Total-spawn ceiling** — bound the total agents a run may create; a runaway loop is a bug.
- **Gauge cost on a slice first** — run one representative unit before fanning out the whole set; extrapolate cost and fix the brief before committing to N.
- **Isolated copy per parallel edit** — if agents mutate files concurrently, give each its own worktree/copy so they cannot corrupt each other (see `using-git-worktrees`).
- **Adversarial cross-check** — route each finding to an independent verifier and report only survivors (see `code-review-patterns` debate mode). A finding one agent produced and the same agent confirmed is not verified.

### Choosing an execution mode

| Question | Inline (this context) | Subagents (dispatch) | Deterministic workflow |
|----------|----------------------|----------------------|------------------------|
| Who holds the plan? | You | You (thin orchestrator) | The script |
| Where do results live? | Conversation | Files (briefs, findings) | Structured returns |
| Repeatable / resumable? | No | Partly | Yes (same script + args) |
| Scale | A few steps | Tens of tasks | Hundreds, pipelined |

Reach for a workflow when control flow should be deterministic (loops, fan-out, verify gates); dispatch when judgment-heavy tasks are independent; stay inline when the work is small or tightly coupled.

## Integration

**Pairs with:** `subagent-driven-development` (run independent plan tasks concurrently)
**Uses:** `verification-before-completion` for the evidence hierarchy when weighing what an agent reports
**See also:** `.claude/CLAUDE.md` Execution Model
