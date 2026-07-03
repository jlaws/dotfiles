---
name: dispatching-parallel-agents
description: "Use when several independent problems can be worked concurrently — dispatches one focused subagent per problem domain in a single message. Do NOT use for interconnected failures, work that needs whole-system context, or tasks that edit overlapping code (keep those in one context)."
skills:
  - verification-before-completion
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

**Multiple subagent calls in a single message run in parallel.** Sequential messages do not.

Each brief must be:

- **Focused** — one test file, one subsystem, one problem
- **Self-contained** — all background the agent needs; it can't see your context
- **Explicit about constraints** — what it must NOT change
- **Specific about the deliverable** — exact output format (findings file, patch, summary)

Hand large inputs over as files (see `subagent-driven-development` handoffs), not inline text.

## Post-Run Integration

Parallel agents finishing is not the end. After they return:

1. **Check for conflicting edits** — did two agents touch the same file? Reconcile.
2. **Review each summary against source** — a subagent reporting success is not proof (`verification-before-completion`). Spot-check the diff.
3. **Run the full suite** — parallel fixes that each pass in isolation can still break in combination.

## Red Flags

- Dispatching interconnected problems in parallel (findings contradict, fixes fight)
- Briefs that assume the agent shares your context
- No stated constraint, so agents wander outside scope
- Trusting parallel summaries without checking the combined diff + full suite
- Parallelizing work that edits overlapping files

## Integration

**Pairs with:** `subagent-driven-development` (run independent plan tasks concurrently)
**Uses:** `verification-before-completion` to validate each agent's output
**See also:** your agent configuration's Execution Model
