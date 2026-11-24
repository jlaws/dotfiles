# Task: Stop kbeval run from measuring a stale staged plugin

## Context
`python3 -m evals.kbeval run --plugin <dir>` runs cases against a directory that `make eval-stage`
wrote earlier. A review of the eval harness on 2026-11-23 found nothing ties that directory to the
working tree.

## Problem
- `run --plugin` (`evals/kbeval/__main__.py:46-50,133`) never restages, so an edit made after
  `make eval-stage` is measured as the old text with no warning.
- A fix cannot restage in place: `stage()` deletes its target (`evals/kbeval/stage.py:53-56`), so a
  second invocation would remove the plugin a running one is reading.

## Scope
1. Stage per invocation into its own directory, or refuse a staged plugin older than the tree.
