# Task: Fix KB eval case defects found in the first baseline

## Context
Reading the raw per-run results of the 2026-11-16 baseline (KB 479cc63) for the asset
determination (`evals/README.md`, Asset determination) turned up cases that score the case, not the
asset. Three grader bugs were fixed in that work; the cases on the original list were fixed by the
eval case audit (`evals/README.md`, Case audit). What remains needs the next baseline.

## Problem
- Two fixed cases need the next baseline to confirm the fix: verification-before-completion
  mocked-green and using-git-worktrees isolated-from-origin both failed in both arms.
- code-review-patterns--effect-receiving-comments now grades only `replies.md` (stop and ask before
  implementing any comment). A run that edits `calc/ops.py` and still writes that nothing is
  implemented passes: lint's grounding rule bars a grader that passes on the unchanged seeded file.
  The builder's "do not ask questions" line also pulls against the rule; the next baseline shows
  whether either arm writes the question.

## Scope
1. Read the next baseline's rows for the cases named above and close each entry it confirms.
