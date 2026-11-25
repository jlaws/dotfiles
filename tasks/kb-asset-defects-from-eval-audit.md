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
  review task its description should catch), on 2026-11-16 and again in the 2026-12 Sonnet
  baseline (0 of 3). Every kb run then called the login shortcut sanctioned, which the skill's
  "Never a security control" line forbids. The case is sound; the description's reach is the
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
- `references/languages/uv-workflows.md` teaches `uv sync --frozen` as the CI flag that fails on a
  stale lockfile (its "MUST use --frozen in CI" block and its "`--frozen` in CI" bullet). In uv
  0.13.8, `--frozen` syncs without updating `uv.lock` and `--locked` is the flag that asserts the
  lockfile is unchanged. j-lang--effect-uv-project-notes requires `--locked`, so a run that
  follows the reference fails it. Both reference trees carry the text.
- `references/languages/python-packaging-and-distribution.md` shows `"pydantic>=2.0,<3.0"` with
  the comment "Cap only at known-breaking major versions", a cap on an unreleased major, against
  its own gotcha "only cap at known-breaking versions". Minor.
- `references/languages/js-ts-patterns.md` shows a library `exports` map with `"import"` before
  `"types"`. TypeScript reads conditions in order, so `"types"` has to come first or the type
  entry is never reached. Both reference trees carry the example.
- `references/product/scope-review-methodology.md` defines EXPAND as "What's the 10x version?"
  for greenfield work; `.claude/agents/scope-reviewer.md` defines it as "the proposed scope is
  missing capabilities required for the stated goal". The agent reads the reference, so the two
  definitions meet in one run.
- `references/cloud/file-storage-patterns.md` gives the presigned-expiry rule for a whole-file
  URL only ("set generous expiry (1h+) for large files") and says nothing about multipart part
  URLs. In the 2026-12 baseline most runs signed short-lived part URLs on demand, and one kb run
  read the rule and set it aside as not applying to parts. The reference should say what it wants
  for part URLs.

## Scope
1. Fix each asset with an eval case that fails before the fix.
