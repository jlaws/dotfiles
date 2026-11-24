# Task: Fix KB eval case defects found in the first baseline

## Context
Reading the raw per-run results of the 2026-11-16 baseline (KB 479cc63) for the asset
determination (`evals/README.md`, Asset determination) turned up cases that score the case, not the
asset. Three grader bugs were fixed in that work; the cases on the original list were fixed by the
eval case audit (`evals/README.md`, Case audit), and the 2026-12 baseline confirmed the two fixes
it was waiting on (verification-before-completion mocked-green: Sonnet 0.83 in both arms;
using-git-worktrees isolated-from-origin: Sonnet kb 1.00, vanilla 0.22).

## Problem
- code-review-patterns--effect-receiving-comments grades only `replies.md` (stop and ask before
  implementing any comment). A run that edits `calc/ops.py` and still writes that nothing is
  implemented would pass: lint's grounding rule bars a grader that passes on the unchanged seeded
  file. No run in the 2026-12 Sonnet baseline did that (all 6 asked the question and said they had
  implemented a comment, so all 6 failed), and the builder's "do not ask questions" line did not
  stop the question.
- using-git-worktrees--effect-land-uncommitted-work: 4 of 6 Sonnet runs in the 2026-12 baseline
  stopped after 3 turns with no file written. Results keep no transcript, so the cause is unknown;
  finding it needs a `--keep-temp` run of that case.
- using-git-worktrees--effect-land-uncommitted-work `merged-not-copied` requires
  `Merge[^\n]*feat/rst`, so a real merge commit with a custom message would fail it. No stored run
  has hit this.

## Scope
1. Probe the worktree case with a kept trace and fix whichever of the case or the skill stops the
   run.
2. Loosen `merged-not-copied` to accept any merge commit that has `feat/rst` as a parent.
