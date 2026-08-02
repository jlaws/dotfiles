---
name: writing-plans
description: "Use when planning a multi-step implementation before coding."
allowed-tools: Read, Grep, Glob, Bash
---

# Writing Plans

## Overview

Write comprehensive implementation plans assuming the engineer has zero codebase context. Document everything they need: which files to touch, complete code samples, how to test, exact commands with expected output. Bite-sized tasks. DRY. YAGNI. TDD. Frequent commits.

Target audience: skilled developer unfamiliar with your codebase and toolset.

**Where the plan lives:** the harness's native plan storage, not the repository. A plan is working state for the session that implements it — it is not a shipped artifact and is not committed.

- **Claude Code:** the plan file assigned by plan mode. Outside plan mode, create `~/.claude/plans/<repo-name>-<feature>.md` — that directory is flat and shared across every project, so the repo name is what keeps plans distinguishable.
- **Other harnesses:** keep the plan in context. Offer to save it to a temp path or a gitignored location if the user wants it durable. Never commit it unless the user asks.

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

## Mandatory Phase Skeleton

Phases are the single unit of grouping — a plan is phases, a phase is tasks. Every plan uses this order. The ends are fixed; the middle expands to as many implementation phases as the work needs.

Each phase carries a goal, an acceptance check, and its own doc delta:

```markdown
## Phase 3: Retrieval rewrite

**Goal:** top-k recall above 0.9 on the eval set
**Acceptance:** `make gate` -> PASS
**Docs:** README retrieval section, `docs/api/search.md`

### Task 1: Write the failing recall test ...
### Task 2: Implement ...
### Task 3: Commit
```

A phase may also be a **decision gate** — prototype, measure, then choose. State the measurement and the branch it decides:

```markdown
## Phase 2: Evaluate cache strategy (prototype)

**Goal:** determine whether Redis or an in-memory LRU meets the 5 ms requirement
**Decision gate:** measure p99 after Task 5. If LRU exceeds 5 ms, Phase 3 uses Redis.
```

After a decision-gate phase the executor pauses and reports findings before continuing.

### Phase 0 — Branch hygiene

No plan starts on `main`. The first phase of every plan is:

```bash
git fetch origin main
git status --porcelain              # must be empty; stop and report if not
git checkout -b <type>/<short-description> origin/main
```

Branch naming follows `type/short-description`. If the plan spans multiple PRs, every later PR repeats this from a clean slate — see PR Boundaries below.

### Phase 1 — Documentation

Write the docs and READMEs describing the behavior the plan will create, before the code exists. The docs are the contract the implementation then satisfies. Name exact files; never write "update the docs".

### Phases 2..N-1 — TDD implementation

Each follows `test-driven-development`: write the failing test, run it and confirm it fails for the right reason, write the minimal implementation, confirm green, commit.

**Each of these phases leads with the doc delta for its own work, then the code.** Phase 1 sets the contract; later phases correct it wherever reality diverged. A phase that changes documented behavior without touching the doc in the same phase is incomplete.

Order them **risk-first**: the hardest or least-certain work goes first, so a wrong assumption surfaces while the plan is still cheap to change. Make each phase a **vertical slice** that produces something testable end to end, rather than a horizontal layer that proves nothing until the layer above it lands.

### Phase N — Validation gate

Prove the capability end to end, not just that the unit tests pass. State one of:

```
Benchmark: bench/retrieval_recall.py::test_top_k
Gate command: make gate  ->  expect PASS
```

```
Benchmark: N/A -- repo has no benchmark harness; the capability is validated by
`make verify` plus the Phase 3 acceptance check.
```

Extend an existing benchmark where one covers the surface; add a new one where none does. An unstated benchmark decision is a failed plan — silence is not N/A.

### PR Boundaries

Commit after every phase. Open a PR at the end, when the work is ready for review.

**Split into multiple PRs when the change is substantial.** Each PR must land the tree in a valid state: its doc updates and its code updates ship together, so `post-ship-doc-sync` and a diff review find nothing stale. After each PR, **wait for review** before starting the next.

Between PRs, reset to a clean slate:

```bash
git checkout main
git fetch origin main
git reset --hard origin/main
git branch --merged origin/main | grep -vE '^\*|^\s*(main|master)$' | xargs -r git branch -d
git checkout -b <type>/<next-description> origin/main
```

A plan that spans PRs states its boundaries explicitly: which phases belong to which PR, and what each PR is reviewable for.

### Pre-production

Assume the system is pre-production unless the plan says otherwise. Breaking changes are acceptable — prefer the clean design over a compatibility shim, and say so in the plan rather than leaving the reader to guess.

## Bite-Sized Task Granularity

Each step is one action (2-5 minutes):

- "Write the failing test" — step
- "Run it to make sure it fails" — step
- "Implement the minimal code to make the test pass" — step
- "Run the tests and make sure they pass" — step
- "Commit" — step

If a step takes more than 5 minutes, split it further.

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
- **Resolve all ambiguities** — no "choose appropriate X" or "use a suitable library". Every decision is made in the plan. If you can't decide, flag it as a decision-gate phase.
- **Documentation task** — if the change alters public surface or documented behavior, include an explicit doc-update task; don't leave docs implicit. See `documentation-validation`.
- **Phase skeleton** — Phase 0 branch hygiene, Phase 1 docs, Phases 2..N-1 TDD, Phase N validation gate.
- **Stated validation gate** — the benchmark and gate command, or an explicit N/A with a reason.
- **Stated PR boundaries** — one PR, or which phases map to which PR with a review wait between.
- **Risk-first vertical slices** — hardest or least-certain phase first; each phase testable end to end rather than a horizontal layer.
- **Decisions persisted** — every significant, not-easily-reversible decision has an ADR task (`docs/adr/NNNN-title.md`); minor ones are noted inline. See `.claude/references/architecture/architecture-decision-records.md`.

## Self-Review (before handoff)

Before presenting execution options, review the finished plan against this checklist and fix any gap:

- **Spec coverage** — every requirement maps to at least one task; nothing dropped.
- **Placeholder scan** — no "TBD", "add validation", "handle edge cases", or "choose appropriate X" remains; every decision is made in the plan.
- **Type/signature consistency** — function and type signatures match across every task that references them.
- **Junior-engineer bar** — the plan is good enough only if an enthusiastic junior engineer with poor taste, no judgement, no project context, and an aversion to testing could execute it correctly. If any step relies on taste or unstated context, make it explicit. Name YAGNI and DRY as constraints where a task invites over-building.
- **Documentation coverage** — a change that ships a public-surface or behavior change has a doc-update task, or the plan states docs are N/A with a reason.
- **Skeleton conformance** — Phase 0 fetches and branches off `origin/main`; Phase 1 is docs; the last phase is the validation gate; every middle phase leads with its own doc delta.
- **Validation gate stated** — a named benchmark and gate command, or an explicit N/A with a reason.
- **Risk order and slicing** — the riskiest work is first, and no phase is a horizontal layer that proves nothing on its own.

## Execution Handoff

After the self-review, present execution options:

```
Plan saved to `<plan-file-path>`. Execution options:

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
