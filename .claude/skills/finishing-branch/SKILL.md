---
name: finishing-branch
description: "Use when completed branch work needs integration."
allowed-tools: Read, Grep, Glob, Bash
skills:
  - documentation-validation
---

# Finishing a Development Branch

**Core principle:** Verify tests → Validate docs → Open the PR.

Completed work ends in a pull request. That is the default and it does not need to be asked for — a
finished branch sitting unpushed is work the reviewer cannot see. Merging locally, parking the branch,
or discarding it are things the user asks for by name; see Other Outcomes below.

## The Process

### Multi-Branch Integration

When integrating multiple worktree branches:

1. Create integration branch: `git checkout -b integrate/<description> main`
2. Sequentially merge each branch:
   ```bash
   git merge <branch> --no-edit
   ```
3. Run full test suite on merged result
4. Clean up worktrees: `git worktree remove <path>` for each
5. Proceed to Step 1 below on the integration branch

### Step 1: Verify Tests

Run the project's full test suite before pushing anything.

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

### Step 1b: Validate Documentation

Before pushing, run the `documentation-validation` gate: confirm product docs (README/API/CHANGELOG) and any KB self-docs reflect what this branch changed, or declare N/A with a reason. Stale docs block completion just like failing tests.

### Step 2: Determine Base Branch

```bash
git merge-base HEAD origin/main
```

Report which branch this split from. Only stop to ask if the answer is not `main`/`master` — an unexpected base usually means the branch was cut from other in-flight work, and the PR target matters.

### Step 3: Open the PR

If the repo already has an open PR for this branch, push to it rather than opening a second one.

```bash
git push -u origin <feature-branch>

gh pr create --title "<title>" --body "$(cat <<'EOF'
## Summary
<2-3 bullets of what changed>

## Test Plan
- [x] <verification that ran, and its result>
EOF
)"
```

Write a real body — never `--fill`. Report the PR URL. If in a worktree, keep it; the user may need it for PR revisions.

Then stop. Do not merge the PR, and do not start the next work item — the branch is now waiting on review.

## Other Outcomes

These replace Step 3 only when the user names one. Do not offer them as a menu.

**Merge locally** — for work that genuinely does not need review:

```bash
git checkout <base-branch>
git pull
git merge <feature-branch>
<test command>          # verify tests on the merged result
git branch -d <feature-branch>
```

Clean up the worktree afterward if there is one.

**Keep as-is** — report `Keeping branch <name>.` and stop. No cleanup, no worktree removal.

**Discard** — destructive and unrecoverable, so require typed confirmation:

```
This will permanently delete:
- Branch <name>
- All commits since <base-branch>
- Worktree at <path> (if applicable)

Type 'discard' to confirm.
```

Wait for that exact word, then `git checkout <base-branch>` and `git branch -D <feature-branch>`.

## Red Flags

- Proceeding with failing tests
- Asking what to do with a finished branch instead of opening the PR
- Opening a second PR when the branch already has one
- Merging without verifying tests on the merged result
- Deleting work without typed confirmation
- Force-pushing without explicit user request
- Cleaning up the worktree when the user asked to keep the branch as-is
- Pushing with documentation that doesn't match the branch's changes (run `documentation-validation` first)

## Integration

**Called by:** `executing-plans` (Step 5) and `subagent-driven-development` after the final review
**Pairs with:** `using-git-worktrees` for worktree cleanup — note that worktree agents return on-branch and never call this skill
**Uses:** `documentation-validation` before pushing
