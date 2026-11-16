# Task: Make kbeval summarize reject stale and shadowed results

## Context
`python3 -m evals.kbeval summarize` builds `evals/baselines/` from every keyed result under the
results root. The asset determination (`evals/README.md`, Asset determination) reran assets under a
separate purpose so no summary mixes KB versions; these gaps are why.

## Problem
- `summary._load` takes any keyed result without checking its KB digest (`evals/kbeval/summary.py:37`),
  so results from an older KB mix into a new summary.
- A `<case>.json` rerun is shadowed by `<asset>.json`: non-`@` paths load in sorted order and `-`
  sorts before `.`, so the asset batch loads last and wins (`summary.py:37-44`).
- A deleted asset's results stay on disk and resurrect its rows.
- `firedRate` averages every process grader, behavioral ones such as `delegated` or `read-refs`
  included (`summary.py:119`, through `_arm_score` at `:71-76`), so the README's fired/never-fired split is not by whether the
  asset loaded.

## Scope
1. Record the KB digest in each result and skip results whose digest differs from the staged KB.
2. Load single-case reruns after the asset batch.
3. Skip results for assets that no longer have cases.
4. Split `firedRate` into asset-load graders and behavioral graders.
