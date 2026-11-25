# Task: Make a long kbeval run hold one CLI version and account for interrupts

## Context
During the 2026-12 baseline (`evals/README.md`, Baseline 2026-12) Claude Code updated itself from
2.1.331 to 2.1.332 in the middle of a step, though the run's shell had `DISABLE_AUTOUPDATER=1`. The
run was stopped by hand, nine batches were moved aside by hand, and the run was resumed on a pinned
binary.

## Problem
- The runner reads `claude --version` once per step (`runner.Batches`), and writes that value to
  every ledger row and into the batch cache key. Result files record the version that really ran,
  so `summarize` refuses a mix, but only after the whole run has been paid for, and the ledger
  shows the wrong version for the batches after an update.
- Nothing pins the CLI for the length of a run. A run of several hours spans an update window.
- An interrupted batch keeps its full reservation in the ledger (60.00 USD here for
  html-report-authoring) with no way to settle it to the actual cost, so stop-and-resume inflates
  recorded spend and eats into caps.
- A rerun of a batch has the same purpose and batch name as the run it replaces, so superseded
  spend can be found only by looking for duplicate batch names.
- Moving results aside is manual. The runner already quarantines a result that holds a key; it has
  no quarantine for a version mismatch, and none writes a note of what was moved and why.

## Scope
1. Check the version before each batch and stop the run when it differs from the step's.
2. Resolve the `claude` binary once at run start and call that path for every batch.
3. Give the ledger a way to release or settle a reservation left by an interrupt.
4. Record in a ledger row whether it replaces an earlier row for the same batch.
5. Quarantine a batch whose result version differs from the step's, with a note file.
