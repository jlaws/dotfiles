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
- `docs/adr/workflow/side-effect-commands-are-user-invoked.md` (Consequences) says Claude reaches
  j-defrag logic through `refactoring-and-debt`; that skill fired in none of its 18 trigger-should
  or effect kb runs.

## Scope
1. Rewrite each listed prompt or grader so the prompt states what the grader checks.
2. Make feature-order's `order` accept the lookup skill listed first, or state the intended order.
3. Add a lock file to the marker-audit fixture, or drop the claim that needs one.
4. Rework or retire the two cases that fail in both arms.
5. Correct the side-effect ADR's reach-path consequence, or record why it still holds.
