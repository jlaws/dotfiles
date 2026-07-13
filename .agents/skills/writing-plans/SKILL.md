---
name: writing-plans
description: "Structured methodology for writing implementation plans with bite-sized tasks, exact file paths, and TDD integration. Use when you have a spec or requirements for a multi-step task and need to create a detailed plan before writing code. Do NOT use for simple single-file changes or bug fixes."
---

# Writing Plans

## Overview

Write comprehensive implementation plans assuming the engineer has zero codebase context. Document everything they need: which files to touch, complete code samples, how to test, exact commands with expected output. Bite-sized tasks. DRY. YAGNI. TDD. Frequent commits.

Target audience: skilled developer unfamiliar with your codebase and toolset.

**Save plans to:** `docs/plans/YYYY-MM-DD-<feature-name>.md`

## Plan Document Header

Every plan MUST start with:

```markdown
# [Feature Name] Implementation Plan

**Purpose:** [Behavioral statement — what changes for the user/system]

BAD: "Implement caching layer"
GOOD: "After this change, repeated API calls for the same resource return cached results within 5ms instead of hitting the database"

**Architecture:** [2-3 sentences about approach]

**Tech Stack:** [Key technologies/libraries]

**Codebase Orientation:**
- Entry point: `path/to/main.ext`
- Key modules: `path/to/relevant/` — [what it does]
- Test runner: `command` (run from `directory/`)
- Config: `path/to/config` — [relevant settings]

---
```

## Bite-Sized Task Granularity

Each step is one action (2-5 minutes):

- "Write the failing test" — step
- "Run it to make sure it fails" — step
- "Implement the minimal code to make the test pass" — step
- "Run the tests and make sure they pass" — step
- "Commit" — step

If a step takes more than 5 minutes, split it further.

## Milestone Grouping

*Optional — recommended for plans with 6+ tasks.*

Group related tasks into milestones. Each milestone is a narrative arc: goal → work → result → proof.

```markdown
## Milestone 1: Database Foundation

**Goal:** Schema and migrations run cleanly on a fresh database
**Acceptance test:** Run `./bin/rails db:reset` then `./bin/rails db:migrate:status` — shows all migrations "up"

### Task 1: Create schema file ...
### Task 2: Write migration ...
### Task 3: Verify migration ...
```

**Prototyping milestones** — use when the plan has a decision gate:
```markdown
## Milestone 2: Evaluate Cache Strategy (prototype)

**Goal:** Determine whether Redis or in-memory LRU meets the <5ms requirement
**Decision gate:** After Task 5, measure p99 latency. If >5ms with LRU, switch to Redis for Milestone 3.
```

After a prototyping milestone, the executor pauses and reports findings before continuing.

## Task Structure

````markdown
### Task N: [Component Name]

**Behavioral check:** [Observable outcome when this task is done]

BAD: "Database layer is implemented"
GOOD: "`./bin/rails runner 'puts User.create!(name: \"test\").id'` prints an integer"

**Files:**
- Create: `exact/path/to/file.ext`
- Modify: `exact/path/to/existing.ext:123-145`
- Test: `tests/exact/path/to/test.ext`

**Step 1: Write the failing test**

```language
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

**Step 2: Run test to verify it fails**

Run: `test-command path/to/test::test_name`
Expected: FAIL with "function not defined"

**Step 3: Write minimal implementation**

```language
def function(input):
    return expected
```

**Step 4: Run test to verify it passes**

Run: `test-command path/to/test::test_name`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/path/test.ext src/path/file.ext
git commit -m "feat: add specific feature"
```
````

## Requirements

- **Exact file paths** — always, no "add to the appropriate file"
- **Complete code** — paste actual code, not "add validation here"
- **Exact commands** — with expected output, not "run the tests"
- **TDD integration** — each task = failing test → verify fail → implement → verify pass → commit
- **Self-contained tasks** — each task can be understood and executed independently
- **Frequent commits** — one commit per task or logical unit
- **Idempotent steps** — every step safely re-runnable. `CREATE TABLE IF NOT EXISTS`, not `CREATE TABLE`. `mkdir -p`, not `mkdir`. If a step fails midway, re-running it from the top must not corrupt state.
- **Resolve all ambiguities** — no "choose appropriate X" or "use a suitable library". Every decision is made in the plan. If you can't decide, flag it as a decision gate in a prototyping milestone.
- **Documentation task** — if the change alters public surface or documented behavior, include an explicit doc-update task; don't leave docs implicit. See `documentation-validation`.

## Self-Review (before handoff)

Before presenting execution options, review the finished plan against this checklist and fix any gap:

- **Spec coverage** — every requirement maps to at least one task; nothing dropped.
- **Placeholder scan** — no "TBD", "add validation", "handle edge cases", or "choose appropriate X" remains; every decision is made in the plan.
- **Type/signature consistency** — function and type signatures match across every task that references them.
- **Junior-engineer bar** — the plan is good enough only if an enthusiastic junior engineer with poor taste, no judgement, no project context, and an aversion to testing could execute it correctly. If any step relies on taste or unstated context, make it explicit. Name YAGNI and DRY as constraints where a task invites over-building.
- **Documentation coverage** — a change that ships a public-surface or behavior change has a doc-update task, or the plan states docs are N/A with a reason.

## Execution Handoff

After the self-review, present execution options:

```
Plan saved to `docs/plans/<filename>.md`. Execution options:

1. **Execute now (inline)** — work through tasks in batches with review checkpoints
   (uses executing-plans skill; run via /j-execute-plan)

2. **Execute via subagents** — fresh agent per task with per-task spec + quality review
   (uses subagent-driven-development skill; best for large or independent-task plans)

3. **Execute in new session** — open a new session and load executing-plans
   (fresh context per batch)

4. **Manual** — you execute the plan yourself

Which approach?
```

## Common Mistakes

| Bad | Good |
|-----|------|
| "Add validation" | Complete validation code with specific checks |
| "Update the tests" | Exact test code with expected assertions |
| "Run the test suite" | `npm test -- --grep "feature"` → Expected: 5 pass |
| "Modify the config" | Exact config changes with file path and line numbers |
| Tasks that take 30+ min | Split into 2-5 minute steps |
| Assuming reader knows the codebase | Explain where things are and why |
| "Choose an appropriate cache strategy" | "Use `lru-cache` with TTL=300s, max=1000 entries" |
| "Task complete when module works" | Behavioral check with exact command + expected output |
| `CREATE TABLE users (...)` | `CREATE TABLE IF NOT EXISTS users (...)` |
| "Implement the `Widget` abstraction" | Specify exact methods, signatures, return types |
| "Set up the ORM" | Name the library, version, and config file path |
