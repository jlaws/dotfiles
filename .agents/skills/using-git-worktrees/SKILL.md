---
name: using-git-worktrees
description: "Git worktree creation for isolated feature work with smart directory selection and safety verification. Use when starting feature work that needs isolation, working on parallel branches, working on multiple features simultaneously, or before executing implementation plans. Do NOT use for general git workflow (check Git Workflow section in CLAUDE.md)."
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
2. ~/.config/superpowers/worktrees/<project-name>/ (global)
```

## Safety Verification

**For project-local directories: MUST verify ignored before creating.**

```bash
git check-ignore -q .worktrees 2>/dev/null
```

**If NOT ignored:** Add to .gitignore, commit, then proceed.

**For global directory (~/.config/superpowers/worktrees):** No verification needed.

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

Before returning or signaling completion:

1. **Stage and commit** all changes (nothing untracked or modified)
2. **Squash** into a single commit (three separate Bash tool calls):
   ```bash
   git add -A
   ```
   ```bash
   git reset --soft $(git merge-base HEAD main)
   ```
   ```bash
   git commit -m "<summary of changes>"
   ```
3. **Report** your branch name and worktree path to the parent/caller
4. Do NOT remove the worktree, merge to main, or invoke `finishing-branch`

> The parent agent is responsible for `git merge` and `git worktree remove`.

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
