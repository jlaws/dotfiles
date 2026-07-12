---
name: j-finalize-pr
description: "Finalize a feature branch's PR — fetch origin/main, squash to one commit, rebase onto main, run tests, force-push with lease, and open or update the PR (stays on the branch). Use when a branch is ready to publish or refresh against latest main. Do NOT use on main/master, or for the initial simple PR without a rebase (use /j-create-pr)."
argument-hint: "[pr title or description]"
model: opus
---

Finalize the current feature branch's PR: pull in the latest `main`, collapse the branch to one clean commit, rebase onto `main`, verify, force-push, and open or update the PR. Stay on the branch — do NOT merge to main or clean up.

PR title/description: $ARGUMENTS

If no arguments, derive the commit message and PR title from the branch's commit log.

---

## Phase 1: Preflight & Safety

1. `git branch --show-current` -> capture as `BRANCH`. If it is `main` or `master`, ABORT immediately — this command never rewrites the trunk.
2. `git status --porcelain` — require a clean working tree with all changes committed. If anything is uncommitted or staged, STOP and tell the user to commit first. Do NOT auto-stash.
3. Establish the base: the remote trunk is `origin/main`. Capture `BRANCH` for later use.

## Phase 2: Fetch & Squash

1. `git fetch origin main` — get the latest trunk.
2. Squash every branch commit into one, based at the pre-rebase merge-base (repo's canonical 3-call sequence — run each as a separate command):
   ```bash
   git add -A
   ```
   ```bash
   git reset --soft $(git merge-base HEAD origin/main)
   ```
   ```bash
   git commit -m "<message>"
   ```
   Use `$ARGUMENTS` as the commit message if provided (imperative, <72-char subject); otherwise summarize `git log --oneline $(git merge-base HEAD origin/main)..HEAD` into one imperative subject.

Squash first, then rebase — a single commit resolves conflicts once instead of per-commit, and matches the requested order.

## Phase 3: Rebase onto main

1. `git rebase origin/main`.
2. On conflict: resolve by integrating `main`'s changes into the branch's code so the new code works against the updated trunk (this is the whole point — propagate main's changes forward). Then `git add <files>` and `git rebase --continue`.
3. If a conflict cannot be resolved with confidence, run `git rebase --abort` and hand the branch back untouched with an explanation. Never guess at a resolution.

## Phase 4: Verify Gate

Detect and run the repo's checks before pushing:
- If a `Makefile` exposes them, run `make check` then `make test`.
- Otherwise detect the toolchain (`package.json` scripts, `pytest`, `cargo test`, etc.) and run its test/build/lint.

STOP before pushing on any failure. A rebase can introduce a semantic break even with no textual conflict — a broken branch must not be published. Report the failure and let the user fix it.

## Phase 5: Force-push

1. Guard: re-confirm `BRANCH` is not `main`/`master`.
2. `git push --force-with-lease origin "$BRANCH"` — `--force-with-lease` refuses to clobber remote commits you have not seen. Never plain `--force`, never to trunk.

## Phase 6: Open or Update the PR

1. Detect an existing PR: `gh pr view --json number,url,title`.
2. If a PR exists: the force-push already updated its head — report the URL. Run `gh pr edit --title "..."` only if `$ARGUMENTS` implies a new title.
3. If no PR exists: create one with a real body (never `--fill`), derived from `git log origin/main..HEAD`:
   ```bash
   gh pr create --title "<title>" --body "$(cat <<'EOF'
   ## Summary
   <2-3 bullets of what changed>

   ## Test Plan
   - [ ] <verification steps>
   EOF
   )"
   ```
4. Stay on `BRANCH`. Report the PR URL, the commit count (1), and files changed.

---

### Cross-References

- **agent:create-pr** -- baseline stage/commit/branch/push/open logic this command extends with fetch, rebase, and force-push
- **skill:finishing-branch** -- base detection via `git merge-base` and the heredoc PR body; note it flags force-push, which is intentional and authorized here
- **skill:using-git-worktrees** -- source of the canonical 3-call squash sequence
