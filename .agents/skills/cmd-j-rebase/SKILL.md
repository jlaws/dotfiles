---
name: cmd-j-rebase
description: "Use when invoking the j-rebase workflow."
disable-model-invocation: true
---

# Rebase

Collapse the current branch to one commit, rebase it onto the latest `origin/main`, verify the merged result with targeted tests, and force-push to the open PR. Stay on the branch — do NOT merge to main or clean up.

Squash commit message: the user's provided input. If none, derive the message from the branch's commit log.

Run end to end without asking for confirmation. The only exits are the hard gates below.

## Phase 1: Preflight

Check all three before touching history, so a failed precondition leaves the branch exactly as it was.

1. `git branch --show-current` — capture as `BRANCH`. If it is `main` or `master`, ABORT — this command never rewrites the trunk.
2. `git status --porcelain` — require a clean working tree with everything committed. If anything is uncommitted or staged, STOP and tell the user to commit first. Do NOT auto-stash.
3. `gh pr view --json number,url,state` — require an open PR for `BRANCH`. If there is none, do nothing at all: no fetch, no squash, no rebase, no push. Report that the branch has no PR and stop.

## Phase 2: Fetch & Squash

1. `git fetch origin main` — get the latest trunk.
2. `git rev-parse HEAD` — record this as the pre-rewrite recovery point and include it in the final report.
3. Squash every branch commit into one, based at the pre-rebase merge-base (canonical 3-call sequence — run each as a separate command):
   ```bash
   git add -A
   ```
   ```bash
   git reset --soft $(git merge-base HEAD origin/main)
   ```
   ```bash
   git commit -m "<message>"
   ```
   Use the provided input as the commit message if given (imperative, <72-char subject); otherwise summarize `git log --oneline $(git merge-base HEAD origin/main)..HEAD` into one imperative subject.

Squash first, then rebase — a single commit resolves each conflict once instead of per commit.

## Phase 3: Rebase onto main

1. `git rebase origin/main`.
2. On conflict, integrate rather than pick a side. Every behavior from the squashed commit must survive, rewritten to work against the updated trunk — this is the whole point. Taking `--ours` or `--theirs` wholesale is a failure, not a resolution. Re-read each resolved file end to end and confirm the branch's intent is intact, then `git add <files>` and `git rebase --continue`.
3. Even with zero textual conflicts, inspect `git diff origin/main...HEAD` for semantic drift — the branch may call APIs that `main` renamed, moved, or deleted.
4. If a conflict cannot be resolved with confidence, run `git rebase --abort` and hand the branch back untouched with an explanation. Never guess at a resolution.

## Phase 4: Targeted Verify Gate

1. `git diff --name-only origin/main..HEAD` — the merged file set.
2. Run only the tests that directly cover those files. Do NOT run the full suite, a whole-repo lint, or a full build — the point is a fast gate on what actually changed.
3. If a changed file has no covering test, name it in the report rather than passing over it silently.
4. Apply the `documentation-validation` gate: confirm product docs and any KB self-docs match this branch's changes, or declare N/A with a reason. Passing tests do not prove docs are current.

STOP before pushing on any failure. A rebase can introduce a semantic break with no textual conflict, and a broken branch must not reach the PR. Report the failure and let the user fix it.

## Phase 5: Force-push

1. Guard: re-confirm `BRANCH` is not `main`/`master`.
2. `git push --force-with-lease origin "$BRANCH"` — `--force-with-lease` refuses to clobber remote commits you have not seen. Never plain `--force`, never to trunk.

## Phase 6: Refresh the PR & Report

1. Update the PR text so it describes the squashed commit rather than the old commit series:
   ```bash
   gh pr edit --title "<squash subject>" --body "$(cat <<'EOF'
   ## Summary
   <2-3 bullets of what changed>

   ## Test Plan
   - [x] <targeted tests that ran, and their result>
   EOF
   )"
   ```
   Write a real body — never `--fill`. Carry over any reviewer-relevant detail from the old body that still applies.
2. Stay on `BRANCH`. Report the PR URL, the commit count (1), files changed, which targeted tests ran with their result, and the Phase 2 recovery SHA.

### Cross-References

- **agent:create-pr** -- baseline stage/commit/push/open logic this command extends with fetch, rebase, and force-push
- **skill:finishing-branch** -- base detection via `git merge-base` and the heredoc PR body; note it flags force-push, which is intentional and authorized here
- **skill:using-git-worktrees** -- source of the canonical 3-call squash sequence
- **skill:documentation-validation** -- per-change doc gate applied in Phase 4 before force-push
