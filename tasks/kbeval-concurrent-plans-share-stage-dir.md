# Task: Keep two kbeval plans from restaging one directory

## Context
`make eval-pilot`, `eval-smoke`, and `eval-baseline` each restage the plugins at start through
`orchestrate._stage`, which stages into a fixed directory under the state root:
`stage/<staging>` (plus a `-x-<tags>` suffix when tags are excluded). Staging into a fixed path is
deliberate: `seed.sh` holds the staged path, so a new directory per invocation would re-key every
cached batch (ADR `kb-evals-run-on-a-staged-plugin`, Ruled Out).

## Problem
- `stage()` removes its target before it rebuilds it. A second plan started while the first is
  running replaces the plugin directory the first is reading, so the first plan's later batches run
  against files that are half rebuilt or built from a newer tree.
- Nothing locks the directory or tells the second invocation that another plan is using it.
- The stale-plugin refusal on `run --plugin` does not help: the marker matches the second plan's
  tree, and the first plan never re-checks it.

## Scope
1. Detect a second plan on the same staging directory and refuse it, or serialize plans with a lock
   held for the plan's duration.
2. Keep the staged path stable, so cached batches stay valid.
