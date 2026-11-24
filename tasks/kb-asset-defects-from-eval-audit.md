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
- `.claude/commands/j-plan.md` (Step 0 and Step 6) has no path for a session without plan mode:
  it calls `EnterPlanMode` and `ExitPlanMode`, and the plan may go only in plan mode's file.
  `writing-plans`, `j-execute-plan`, and `j-next` each state an "Outside plan mode" fallback. The
  eval harness has no plan mode, so the j-plan cases now name a git-ignored plan file.
- `.claude/commands/j-next.md`: Phase 2 step 6 edits the plan file, then Phase 3 runs
  `git reset --hard origin/main`. When the plan file is tracked in the repo, the reset discards
  that edit. Low severity: plans normally live under `~/.claude/plans/`. Seen in
  j-next--effect-second-part.
- `references/languages/uv-workflows.md` teaches `uv sync --frozen` as the CI flag that fails on a
  stale lockfile (its "MUST use --frozen in CI" block and its "`--frozen` in CI" bullet). In uv
  0.13.8, `--frozen` syncs without updating `uv.lock` and `--locked` is the flag that asserts the
  lockfile is unchanged. j-lang--effect-uv-project-notes requires `--locked`, so a run that
  follows the reference fails it. Both reference trees carry the text.
- `references/languages/python-packaging-and-distribution.md` shows `"pydantic>=2.0,<3.0"` with
  the comment "Cap only at known-breaking major versions", a cap on an unreleased major, against
  its own gotcha "only cap at known-breaking versions". Minor.

## Scope
1. Fix each asset with an eval case that fails before the fix.
