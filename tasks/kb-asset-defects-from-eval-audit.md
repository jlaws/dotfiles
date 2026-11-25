# Task: Fix KB asset defects found by the eval case audit

## Context
The eval case audit (`evals/README.md`, Case audit) checks every case against its asset. Where a
case is sound and the asset is at fault, the case is left alone and the asset defect is listed
here, so the fix can come with an eval case that fails before it.

## Problem
- `code-quality` loaded in none of the stored kb runs of code-quality--effect-marker-audit (a
  review task its description should catch), on 2026-11-16 and again in the 2026-12 Sonnet
  baseline (0 of 3). Every kb run then called the login shortcut sanctioned, which the skill's
  "Never a security control" line forbids. The case is sound; the description's reach is the
  question.

## Scope
1. Fix each asset with an eval case that fails before the fix.
