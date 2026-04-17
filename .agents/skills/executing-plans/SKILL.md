---
name: executing-plans
description: "Structured methodology for executing implementation plans in batches with review checkpoints. Use when you have a written plan to execute task-by-task with verification gates between batches. Do NOT use for ad-hoc implementation without a plan (use design-first then writing-plans first)."
---

# Executing Plans

**Core principle:** Batch execution with checkpoints for review.

## The Process

### Step 1: Load and Review Plan

1. Read the plan file
2. Review critically — identify questions, gaps, or concerns
3. If concerns: raise them with the user before starting
4. If clear: proceed to execution

### Step 2: Execute Batch

**Default batch size: 3 tasks**

For each task in the batch:
1. Announce which task you're starting
2. Follow each step exactly as written in the plan
3. Run all verification commands specified
4. Apply `verification-before-completion` — confirm each step passes before moving on
5. Commit after each task (or as the plan specifies)

#### Living Document Maintenance

**NOT optional.** Maintain these sections in the plan file as you execute:

**Progress** — checklist updated after each task:
```markdown
## Progress
- [x] Task 1: Setup schema — `a1b2c3d`
- [x] Task 2: Create migration — `e4f5g6h`
- [ ] Task 3: Add model layer ← current
- [ ] Task 4: Write API endpoints
```

**Decision Log** — judgment calls when the plan is ambiguous or reality diverges:
```markdown
## Decision Log
| Task | Decision | Rationale |
|------|----------|-----------|
| 3 | Used `jsonb` instead of `json` column type | Plan said "JSON column" — `jsonb` supports indexing, matches existing schema pattern in `users` table |
| 5 | Skipped Redis cache, used in-memory LRU | Redis not in docker-compose; plan's behavioral check (< 5ms response) passes with LRU |
```

**Surprises & Discoveries** — unexpected behaviors with evidence:
```markdown
## Surprises & Discoveries
- Task 2: Migration fails silently when `pgcrypto` extension missing. Fixed by adding `CREATE EXTENSION IF NOT EXISTS pgcrypto;`. Error was: `PG::UndefinedFunction: ERROR: function gen_random_uuid() does not exist`
```

### Step 3: Report

After each batch, present evidence artifacts — not summaries:

- **Paste terminal output** as code blocks. Don't paraphrase test results or build output.
- **Run behavioral acceptance checks** from the plan (milestone acceptance tests, task behavioral checks) and paste results.
- **Include Decision Log entries** made during this batch.
- **Include Surprises & Discoveries** from this batch.
- Say: **"Batch complete. Ready for feedback."**
- Wait for user response before continuing

### Step 4: Continue

Based on feedback:
- Apply requested changes
- Execute next batch of 3 tasks
- Repeat until all tasks complete

### Step 5: Complete

After all tasks are done and verified:
- Run full test suite one final time
- Load `finishing-branch` skill to wrap up the branch
- Follow that skill's process (verify → present options → execute choice)

## When to Stop and Ask

**STOP executing immediately when:**
- A test fails and the fix isn't obvious from the plan
- A dependency is missing or unavailable
- An instruction in the plan is unclear or ambiguous
- The plan's assumptions don't match reality (file doesn't exist, API changed, etc.)
- You've hit 3+ consecutive unexpected issues
- A non-idempotent step partially executed (e.g., half a migration ran). Stop immediately — re-running may corrupt state. Report exactly what happened and what state you're in.

**Ask for clarification rather than guessing.** Don't force through blockers.

## Batch Size Adjustment

- User can request different batch sizes ("do 5 at a time", "one at a time")
- For risky or complex tasks, drop to batch size 1
- For straightforward tasks, batch size can increase to 5

## Progress Tracking

Use the **Progress** section in the plan file (see Living Document Maintenance above) as the single source of truth. When reporting progress inline, narrate with evidence:

```
Completed Task 3 (Add model layer):
- Created `app/models/user.rb` with validations
- Behavioral check passed:
  $ ./bin/rails runner 'puts User.create!(name: "test").id'
  42
- Committed: `a1b2c3d feat: add User model with validations`

Starting Task 4 (Write API endpoints)...
```

## Red Flags

- Skipping verification steps to move faster
- Continuing past a failing test
- Modifying the plan without user approval
- Executing tasks out of order without justification
- Committing broken code between tasks
- Guessing when the plan is unclear
- Summarizing verification output instead of pasting it (paste the actual terminal output)
- Making decisions without logging them (every judgment call goes in the Decision Log)

## Integration

**Receives plans from:** `writing-plans`
**Hands off to:** `finishing-branch` when all tasks complete
**Uses:** `verification-before-completion` for each verification step
