---
name: finishing-branch
description: "Structured 4-step process for completing a development branch — verify tests, determine base, present integration options, execute with cleanup. Use when implementation is complete and you need to decide how to integrate the work. Do NOT use mid-implementation or before tests pass."
---

# Finishing a Development Branch

**Core principle:** Verify tests → Present options → Execute choice → Clean up.

## The Process

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

## Quick Reference

| Option | Merge | Push | Keep Worktree | Delete Branch |
|--------|-------|------|---------------|---------------|
| 1. Merge locally | yes | - | no | yes (safe) |
| 2. Create PR | - | yes | yes | - |
| 3. Keep as-is | - | - | yes | - |
| 4. Discard | - | - | no | yes (force) |

## Red Flags

- Proceeding with failing tests
- Merging without verifying tests on the merged result
- Deleting work without typed confirmation
- Force-pushing without explicit user request
- Offering fewer or more than 4 options
- Cleaning up worktree when user chose "keep as-is"
## Integration

**Called by:** `executing-plans` (Step 5) after all tasks complete
**Pairs with:** `using-git-worktrees` for worktree cleanup
