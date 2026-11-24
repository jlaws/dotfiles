# Task: Fix KB eval case defects found in the first baseline

## Context
Reading the raw per-run results of the 2026-11-16 baseline (KB 479cc63) for the asset
determination (`evals/README.md`, Asset determination) turned up cases that score the case, not the
asset. Three grader bugs were fixed in that work; these need case rewrites or a decision.

## Problem
- Prompts that omit a graded criterion, so a run can fail a claim it was never asked to meet:
  debugging-methodology unreproducible-502, dependency-upgrade multi-major-plan, j-audit
  login-module scope, documentation-writer changelog category.
- skill-lookup-discipline feature-order: `order` fails when the lookup skill lists itself first.
- code-quality marker-audit: the fixture has no lock file.
- verification mocked-green and using-git-worktrees isolated-from-origin fail in both arms, so they
  measure nothing about the asset.
- j-explain `self-contained` (both cases) misses `srcset`, `<object data>`, SVG `<image href>`,
  `fetch`/XHR, and meta refresh, so a report that makes network requests can pass. It also fails a
  self-contained page with a `data:` `<link>` or inline `@import` text. Widening it needs a rerun.
- `docs/adr/workflow/side-effect-commands-are-user-invoked.md` (Consequences) says Claude reaches
  j-defrag logic through `refactoring-and-debt`; that skill fired in none of its 18 trigger-should
  or effect kb runs.
- writing-skills--effect-fix-broken-skill: graders `desc-starts` and `desc` check "Use when" and
  "under 200 characters"; the prompt states neither.
- Suspects to audit for the same flaw (a graded criterion the prompt never states):
  design-first--effect-todo-delete-design, j-skill-audit--effect-full-axes-weakest,
  skill-audit--effect-judgment-pass, skill-audit--effect-mechanical-audit,
  skill-lookup-discipline--effect-bugfix-order.

## Scope
1. Rewrite each listed prompt or grader so the prompt states what the grader checks.
2. Make feature-order's `order` accept the lookup skill listed first, or state the intended order.
3. Add a lock file to the marker-audit fixture, or drop the claim that needs one.
4. Rework or retire the two cases that fail in both arms.
5. Widen j-explain `self-contained` to every network-loading construct, then rerun j-explain.
6. Correct the side-effect ADR's reach-path consequence, or record why it still holds.
