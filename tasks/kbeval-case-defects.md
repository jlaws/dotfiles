# Task: Fix KB eval case defects found in the first baseline

## Context
Reading the raw per-run results of the 2026-11-16 baseline (KB 479cc63) for the asset
determination (`evals/README.md`, Asset determination) turned up cases that score the case, not the
asset. Three grader bugs were fixed in that work; these need case rewrites or a decision. The skill
cases on the original list were fixed by the eval case audit (`evals/README.md`, Case audit).

## Problem
- Prompts that omit a graded criterion, so a run can fail a claim it was never asked to meet: j-audit
  login-module scope, documentation-writer changelog category.
- j-explain `self-contained` (both cases) misses `srcset`, `<object data>`, SVG `<image href>`,
  `fetch`/XHR, and meta refresh, so a report that makes network requests can pass. It also fails a
  self-contained page with a `data:` `<link>` or inline `@import` text. Widening it needs a rerun.
- `docs/adr/workflow/side-effect-commands-are-user-invoked.md` (Consequences) says Claude reaches
  j-defrag logic through `refactoring-and-debt`; that skill fired in none of its 18 trigger-should
  or effect kb runs.
- Suspect to audit for a graded criterion the prompt never states:
  j-skill-audit--effect-full-axes-weakest.
- Two fixed cases need the next baseline to confirm the fix: verification-before-completion
  mocked-green and using-git-worktrees isolated-from-origin both failed in both arms.

## Scope
1. Rewrite each listed prompt or grader so the prompt states what the grader checks.
2. Widen j-explain `self-contained` to every network-loading construct, then rerun j-explain.
3. Correct the side-effect ADR's reach-path consequence, or record why it still holds.
