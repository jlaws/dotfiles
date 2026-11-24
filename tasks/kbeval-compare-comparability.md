# Task: Make kbeval compare check that two summaries are comparable

## Context
`make eval-compare OLD=... NEW=...` reports the change between two baseline summaries. A review of
the eval harness on 2026-11-23 found it compares any two files and flags noise as change.

## Problem
- `compare.compare` (`evals/kbeval/compare.py:27-58`) never checks that the two summaries share a
  judge model and a Claude Code version, so a judge or CLI change reads as a KB change.
- Its `DROP = 1/3` flag fires on 12-34% of unchanged non-ceiling cases at 3 runs, so a "changed
  cases" list is mostly run-to-run variance.

## Scope
1. Refuse, or mark clearly, a comparison whose summaries differ in judge model or Claude Code
   version.
2. Replace or qualify the `DROP` flag so an unchanged case is rarely listed.
