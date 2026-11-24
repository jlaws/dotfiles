# Task: Fix KB asset defects found by the eval case audit

## Context
The eval case audit (`evals/README.md`, Case audit) checks every case against its asset. Where a
case is sound and the asset is at fault, the case is left alone and the asset defect is listed
here, so the fix can come with an eval case that fails before it.

## Problem
- `.claude/skills/using-git-worktrees/SKILL.md` (Safety Verification) says "If NOT ignored: Add to
  .gitignore, commit, then proceed." With staged, uncommitted edits in the tree, a plain
  `git commit` commits them too. It should say to commit only `.gitignore`, or to use
  `.git/info/exclude`. Seen in using-git-worktrees--effect-isolated-from-origin.
- `code-quality` loaded in none of the stored kb runs of code-quality--effect-marker-audit (a
  review task its description should catch). The case is sound; the description's reach is the
  question.
- `dependency-upgrade` states an upgrade order (SKILL.md:12). In the stored kb runs of
  dependency-upgrade--effect-multi-major-plan the skill loaded and the plan still put jest before
  express and mongoose. The case may stay dead for that reason.
- `skill-lookup-discipline`: no stored kb run of skill-lookup-discipline--effect-feature-order
  listed `design-first` for a feature request, though the case expects it first.

## Scope
1. Fix each asset with an eval case that fails before the fix.
