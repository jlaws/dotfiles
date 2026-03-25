---
name: workflow-execute-plan
description: "Execute an implementation plan in batches with review checkpoints. Use when you have a written plan to work through task-by-task. Do NOT use without a plan first (use /write-plan instead)."
argument-hint: "<plan file path>"
---

Plan file: $ARGUMENTS

If no arguments provided, look for the most recent plan in `docs/plans/` and confirm with the user before starting.

---

## Executing Plans

**Core principle:** Batch execution with checkpoints for review.

For implementation, self-review, and review checklists, see `references/workflow/task-execution-checklists`.

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
4. Apply the Verification Before Completion methodology (see below) — confirm each step passes before moving on
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
- Follow the Finishing a Development Branch process (see below) to wrap up the branch
- Follow that process (verify → present options → execute choice)

### When to Stop and Ask

**STOP executing immediately when:**
- A test fails and the fix isn't obvious from the plan
- A dependency is missing or unavailable
- An instruction in the plan is unclear or ambiguous
- The plan's assumptions don't match reality (file doesn't exist, API changed, etc.)
- You've hit 3+ consecutive unexpected issues
- A non-idempotent step partially executed (e.g., half a migration ran). Stop immediately — re-running may corrupt state. Report exactly what happened and what state you're in.

**Ask for clarification rather than guessing.** Don't force through blockers.

### Batch Size Adjustment

- User can request different batch sizes ("do 5 at a time", "one at a time")
- For risky or complex tasks, drop to batch size 1
- For straightforward tasks, batch size can increase to 5

### Progress Tracking

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

### Red Flags

- Skipping verification steps to move faster
- Continuing past a failing test
- Modifying the plan without user approval
- Executing tasks out of order without justification
- Committing broken code between tasks
- Guessing when the plan is unclear
- Summarizing verification output instead of pasting it (paste the actual terminal output)
- Making decisions without logging them (every judgment call goes in the Decision Log)

**Receives plans from:** `workflow/writing-plans`
**Hands off to:** Finishing a Development Branch process (see below) when all tasks complete
**Uses:** Verification Before Completion methodology (see below) for each verification step

---

## Verification Before Completion

**Core principle:** Evidence before claims, always.

**Violating the letter of this rule is violating the spirit of this rule.**

### The Iron Law

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

If you haven't run the verification command in this message, you cannot claim it passes.

### The Gate Function

```
BEFORE claiming any status:
1. IDENTIFY: What command proves this claim?
2. RUN: Execute the FULL command (fresh, complete)
3. READ: Full output, check exit code, count failures
4. VERIFY: Does output confirm the claim?
   - If NO: State actual status with evidence
   - If YES: State claim WITH evidence
5. ONLY THEN: Make the claim
```

### Common Failures

| Claim | Requires | Not Sufficient |
|-------|----------|----------------|
| Tests pass | Test output: 0 failures | Previous run, "should pass" |
| Linter clean | Linter output: 0 errors | Partial check, extrapolation |
| Build succeeds | Build: exit 0 | Linter passing |
| Bug fixed | Original symptom: passes | Code changed, assumed fixed |
| Agent completed | VCS diff shows changes | Agent reports "success" |
| Requirements met | Line-by-line checklist | Tests passing |

### Red Flags - STOP

- Using "should", "probably", "seems to"
- Expressing satisfaction before verification
- About to commit/push/PR without verification
- Trusting agent success reports
- Relying on partial verification
- **ANY wording implying success without having run verification**

### Rationalization Prevention

| Excuse | Reality |
|--------|---------|
| "Should work now" | RUN the verification |
| "I'm confident" | Confidence is not evidence |
| "Just this once" | No exceptions |
| "Linter passed" | Linter is not compiler |
| "Agent said success" | Verify independently |
| "Partial check is enough" | Partial proves nothing |

### Key Patterns

```
Tests:     Run -> See "34/34 pass" -> THEN claim "All tests pass"
Red-Green: Write -> Run (pass) -> Revert -> Run (MUST FAIL) -> Restore -> Run (pass)
Build:     Run build -> See exit 0 -> THEN claim "Build passes"
Requirements: Re-read plan -> Checklist -> Verify each -> Report
Agent:     Agent reports -> Check VCS diff -> Verify changes -> Report actual state
```

### When To Apply

**ALWAYS before:** Any success/completion claim, any positive statement about work state, committing, PR creation, task completion, moving to next task, delegating to agents.

**No shortcuts. Run the command. Read the output. THEN claim the result.**

---

## Finishing a Development Branch

**Core principle:** Verify tests → Present options → Execute choice → Clean up.

### Multi-Branch Integration

When integrating multiple worktree branches:

1. Create integration branch: `git checkout -b integrate/<description> main`
2. Sequentially merge each branch:
   ```bash
   git merge <agent-branch> --no-edit
   ```
3. Run full test suite on merged result
4. Clean up worktrees: `git worktree remove <path>` for each
5. Proceed to Step 1 below on the integration branch

### Step 1: Verify Tests

Run the project's full test suite before presenting any options.

```bash
# Use project-appropriate test command
npm test / cargo test / pytest / go test ./... / swift test
```

**If tests fail:**
```
Tests failing (N failures). Must fix before completing:

[Show failures]

Cannot proceed with merge/PR until tests pass.
```

STOP. Do not proceed to Step 2. Fix tests first.

**If tests pass:** Continue.

### Step 2: Determine Base Branch

```bash
git merge-base HEAD main
```

Confirm with user: "This branch split from `main` — is that correct?"

### Step 3: Present Options

Present exactly these 4 options:

```
Implementation complete. All tests pass. What would you like to do?

1. Merge back to <base-branch> locally
2. Push and create a Pull Request
3. Keep the branch as-is (I'll handle it later)
4. Discard this work
```

Don't add explanation — keep options concise.

### Step 4: Execute Choice

#### Option 1: Merge Locally

```bash
git checkout <base-branch>
git pull
git merge <feature-branch>
# Verify tests on merged result
<test command>
# If tests pass
git branch -d <feature-branch>
```

If in a worktree, clean it up after merge.

#### Option 2: Push and Create PR

```bash
git push -u origin <feature-branch>

gh pr create --title "<title>" --body "$(cat <<'EOF'
## Summary
<2-3 bullets of what changed>

## Test Plan
- [ ] <verification steps>
EOF
)"
```

Report the PR URL. If in a worktree, keep it (user may need it for PR revisions).

#### Option 3: Keep As-Is

Report: "Keeping branch `<name>`. You can return to it later."

No cleanup. No worktree removal.

#### Option 4: Discard

**Require typed confirmation:**
```
This will permanently delete:
- Branch <name>
- All commits since <base-branch>
- Worktree at <path> (if applicable)

Type 'discard' to confirm.
```

Wait for exact confirmation. Then:
```bash
git checkout <base-branch>
git branch -D <feature-branch>
```

Clean up worktree if applicable.

### Quick Reference

| Option | Merge | Push | Keep Worktree | Delete Branch |
|--------|-------|------|---------------|---------------|
| 1. Merge locally | yes | - | no | yes (safe) |
| 2. Create PR | - | yes | yes | - |
| 3. Keep as-is | - | - | yes | - |
| 4. Discard | - | - | no | yes (force) |

### Red Flags

- Proceeding with failing tests
- Merging without verifying tests on the merged result
- Deleting work without typed confirmation
- Force-pushing without explicit user request
- Offering fewer or more than 4 options
- Cleaning up worktree when user chose "keep as-is"
**Called by:** Executing Plans (Step 5) after all tasks complete
**Pairs with:** `workflow/using-git-worktrees` for worktree cleanup
