---
name: using-git-worktrees
description: "Use when isolated branches need separate working trees."
allowed-tools: Read, Grep, Glob, Bash
---

# Using Git Worktrees

**Announce at start:** "I'm using the using-git-worktrees skill to set up an isolated workspace."

## Directory Selection (Priority Order)

### 1. Check Existing Directories
```bash
ls -d .worktrees 2>/dev/null     # Preferred (hidden)
ls -d worktrees 2>/dev/null      # Alternative
```
If both exist, `.worktrees` wins.

### 2. Check CLAUDE.md
```bash
grep -i "worktree.*director" CLAUDE.md 2>/dev/null
```
If preference specified, use it.

### 3. Ask User
```
No worktree directory found. Where should I create worktrees?
1. .worktrees/ (project-local, hidden)
2. ~/.worktrees/<project-name>/ (outside the repo)
```

## Safety Verification

**For project-local directories: MUST verify ignored before creating.**

```bash
git check-ignore -q .worktrees 2>/dev/null
```

**If NOT ignored:** Add to .gitignore, commit, then proceed.

**For a directory outside the repo:** No verification needed.

## Creation Steps

```bash
# 1. Detect project
project=$(basename "$(git rev-parse --show-toplevel)")

# 2. Create worktree
git worktree add "$path" -b "$BRANCH_NAME"
cd "$path"

# 3. Auto-detect and run setup
[ -f package.json ] && npm install
[ -f Cargo.toml ] && cargo build
[ -f requirements.txt ] && pip install -r requirements.txt
[ -f pyproject.toml ] && poetry install
[ -f go.mod ] && go mod download

# 4. Verify clean baseline
# Run project-appropriate test command
# If tests fail: report failures, ask whether to proceed

# 5. Report
# "Worktree ready at <path>, tests passing (N tests, 0 failures)"
```

## Completing Work in a Worktree

This is the contract between a worktree agent and its caller. Each step exists because skipping it
loses work silently.

1. **Commit everything.** Uncommitted changes are invisible to `git merge`, so anything left staged or
   modified is thrown away when the caller integrates.
2. **Squash into one commit**, as three separate Bash calls:
   ```bash
   git add -A
   ```
   ```bash
   git reset --soft $(git merge-base HEAD main)
   ```
   ```bash
   git commit -m "<summary of changes>"
   ```
3. **Report** your branch name and worktree path back to the caller.
4. **Integrate with `git merge`, never by copying files.** `cp` or `rsync` out of a worktree loses
   history and silently overwrites concurrent work in the destination.
5. **Leave the worktree in place.** The caller owns `git merge` and `git worktree remove`, and cannot
   integrate a tree you already deleted. For the same reason, do not invoke `finishing-branch` — return
   the work on its branch.

## Quick Reference

| Situation | Action |
|-----------|--------|
| `.worktrees/` exists | Use it (verify ignored) |
| `worktrees/` exists | Use it (verify ignored) |
| Both exist | Use `.worktrees/` |
| Neither exists | Check CLAUDE.md, then ask user |
| Directory not ignored | Add to .gitignore + commit |
| Tests fail in baseline | Report failures + ask |

## Examples

**Trigger:** "Start isolated feature work without stashing current changes"
**Action:** Create a new git worktree with a feature branch, set up the environment
**Result:** Two independent working directories — original branch untouched, new feature branch ready

## Integration

- **Called by:** brainstorming (after design approved), any skill needing isolation
- **Pairs with:** finishing-a-development-branch (cleanup after), executing-plans (work happens here)
