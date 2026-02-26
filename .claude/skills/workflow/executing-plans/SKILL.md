---
name: executing-plans
description: "Structured methodology for executing implementation plans in batches with review checkpoints. Use when you have a written plan to execute task-by-task with verification gates between batches. Do NOT use for ad-hoc implementation without a plan (use design-first then writing-plans first)."
compatibility: claude-code
allowed-tools: Read, Grep, Glob, Bash
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
4. Apply `workflow/verification-before-completion` — confirm each step passes before moving on
5. Commit after each task (or as the plan specifies)

### Step 3: Report

After each batch:
- Show what was implemented (files changed, features added)
- Show verification output (test results, build status)
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
- Load `workflow/finishing-branch` skill to wrap up the branch
- Follow that skill's process (verify → present options → execute choice)

## When to Stop and Ask

**STOP executing immediately when:**
- A test fails and the fix isn't obvious from the plan
- A dependency is missing or unavailable
- An instruction in the plan is unclear or ambiguous
- The plan's assumptions don't match reality (file doesn't exist, API changed, etc.)
- You've hit 3+ consecutive unexpected issues

**Ask for clarification rather than guessing.** Don't force through blockers.

## Batch Size Adjustment

- User can request different batch sizes ("do 5 at a time", "one at a time")
- For risky or complex tasks, drop to batch size 1
- For straightforward tasks, batch size can increase to 5

## Progress Tracking

Track progress inline:
```
### Task 1: Setup database schema ✅
### Task 2: Create migration ✅
### Task 3: Add model layer ⬜ (current)
### Task 4: Write API endpoints ⬜
```

## Red Flags

- Skipping verification steps to move faster
- Continuing past a failing test
- Modifying the plan without user approval
- Executing tasks out of order without justification
- Committing broken code between tasks
- Guessing when the plan is unclear

## Integration

**Receives plans from:** `workflow/writing-plans`
**Hands off to:** `workflow/finishing-branch` when all tasks complete
**Uses:** `workflow/verification-before-completion` for each verification step
