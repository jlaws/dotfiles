---
name: j-rebase
description: "Squash a branch to one commit, rebase onto latest origin/main, verify the conflict resolutions with targeted tests, and force-push to the open PR. Use when an open PR needs to be refreshed against main. Do NOT use on main/master, or before a PR exists (use /j-create-pr)."
---

# Rebase

Collapse the current branch to one commit, rebase it onto the latest `origin/main`, verify the conflict resolutions with targeted tests, and force-push to the open PR. Stay on the branch -- do NOT merge to main or clean up.

Squash commit message: $ARGUMENTS

If no arguments, derive the message from the branch's commit log.

Run end to end without asking for confirmation. The only exits are the hard gates below.

---

## Phase 1: Preflight

Check all three before touching history, so a failed precondition leaves the branch exactly as it was.

1. `git branch --show-current` -> capture as `BRANCH`. If it is `main` or `master`, ABORT -- this command never rewrites the trunk.
2. `git status --porcelain` -- require a clean working tree with everything committed. If anything is uncommitted or staged, STOP and tell the user to commit first. Do NOT auto-stash.
3. `gh pr view --json number,url,state` -- require an open PR for `BRANCH`. If there is none, do nothing at all: no fetch, no squash, no rebase, no push. Report that the branch has no PR and stop.

## Phase 2: Fetch & Squash

1. `git fetch origin main` -- get the latest trunk.
2. `git rev-parse HEAD` -- record this as the pre-rewrite recovery point and include it in the final report. Also record `git merge-base HEAD origin/main` as `BASE`: the trunk commit the branch was last built on.
3. Squash every branch commit into one, based at the pre-rebase merge-base (repo's canonical 3-call sequence -- run each as a separate command):
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

Squash first, then rebase -- a single commit resolves each conflict once instead of per commit.

## Phase 3: Rebase onto main

1. `git rebase origin/main`.
2. On conflict, integrate rather than pick a side. Every behavior from the squashed commit must survive, rewritten to work against the updated trunk -- this is the whole point. Taking `--ours` or `--theirs` wholesale is a failure, not a resolution. At each conflict stop, before staging anything, add the output of `git diff --name-only --diff-filter=U` to `CONFLICTED`; `git add` clears it. Also add any path git reports as `Staged '<path>' using previous resolution`: rerere already staged it, so the filter misses it. Re-read each resolved file end to end and confirm the branch's intent is intact, then `git add <files>` and `git rebase --continue`. If the rebase finishes without stopping, `CONFLICTED` is empty.
3. Even with zero textual conflicts, check for semantic drift: the branch may call APIs that `main` renamed, moved, or deleted. After the rebase, `git diff origin/main...HEAD` shows only the branch's side, so read what `main` changed with `git diff "$BASE" origin/main` and find the branch code that still depends on it. Fix drift now. Record every branch file that depends on something `main` changed, edited or not, as `DRIFT`; Phase 4 tests it and amends the edits.
4. If a conflict cannot be resolved with confidence, run `git rebase --abort` and hand the branch back untouched with an explanation. Never guess at a resolution.

## Phase 4: Targeted Verify and Repair

1. The test set is `CONFLICTED` plus `DRIFT` from Phase 3. Run the tests that cover those files, plus the tests that cover code importing or calling anything the resolution or a drift fix changed; for a path the resolution deleted, that means its former importers. This verifies the rebase, not the whole branch: the rest of the diff is what the PR already carried. If the test set is empty, run no tests; the final checks in step 7 still run.
2. Do NOT run the full suite, a whole-repo lint, or a full build unless the branch and `main` overlapped significantly. That call is yours: judge it from `CONFLICTED` and `DRIFT`, and when you escalate, state the reason in the report along with the conflicts and drift behind it.
3. If a file in the test set has no covering test, name it in the report rather than passing over it silently.
4. Run the `documentation-validation` gate: confirm product docs and any KB self-docs match this branch's changes, or declare N/A with a reason. Passing tests do not prove docs are current.

5. A failed check blocks the push, not repair work. MUST diagnose and fix recoverable whitespace, formatting, test, and documentation failures within the branch's work without asking for confirmation. An extra blank line introduced during conflict resolution is yours to fix, not a reason to hand work back to the user. Never weaken or skip checks to obtain a pass.
6. Rerun the failed checks and any checks affected by the repair. After two failed attempts on the same error, re-examine the cause and change approach; the retry count alone is not a reason to hand off.
7. If you made repairs or drift fixes, review their diff once checks pass, stage only those files, and run `git commit --amend --no-edit` to retain one commit. In every case, including an empty test set, review the final `git diff origin/main...HEAD`, run `git diff --check origin/main...HEAD`, and confirm a clean working tree and one branch commit. Any new verification failure returns to step 5. When verification passes, continue to Phase 5 and refresh the PR without another confirmation.
8. Stop without pushing only when a genuine blocker requires user input, permissions, or unavailable resources. Report the exact failed command and error, attempted repairs, the concrete blocker, and the Phase 2 recovery SHA. A failed check by itself is not such a blocker.

## Phase 5: Force-push

1. Guard: re-confirm `BRANCH` is not `main`/`master`.
2. `git push --force-with-lease origin "$BRANCH"` -- `--force-with-lease` refuses to clobber remote commits you have not seen. Never plain `--force`, never to trunk.

## Phase 6: Refresh the PR & Report

1. Update the PR text so it describes the squashed commit rather than the old commit series:
   ```bash
   gh pr edit --title "<squash subject>" --body "$(cat <<'EOF'
   ## Summary
   <2-3 bullets of what changed>

   ## Test Plan
   - [x] <`CONFLICTED` and `DRIFT` files, the tests that ran and their result, or "empty test set: no conflicts, no drift">
   EOF
   )"
   ```
   Write a real body -- never `--fill`. Carry over any reviewer-relevant detail from the old body that still applies.
2. Stay on `BRANCH`. Report the PR URL, the commit count (1), files changed, `CONFLICTED`, `DRIFT`, which targeted tests ran with their result and how you found the callers (or that none ran because the test set was empty), whether the full suite ran and why, and the Phase 2 recovery SHA.

---

### Cross-References

- **agent:create-pr** -- baseline stage/commit/push/open logic this command extends with fetch, rebase, and force-push
- **skill:finishing-branch** -- base detection via `git merge-base` and the heredoc PR body; note it flags force-push, which is intentional and authorized here
- **skill:using-git-worktrees** -- source of the canonical 3-call squash sequence
- **skill:documentation-validation** -- per-change doc gate applied in Phase 4 before force-push
