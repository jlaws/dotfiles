# Task: Close the paths where kbeval settles a batch below what it billed

## Context
`runner.execute` reserves a batch's cap, runs the CLI, then settles the reservation to a cost. The
settle path for an interrupt is the `settle` command (see Budget in `evals/README.md`). Three
paths in `execute` still settle to a figure that can be lower than the real spend.

## Problem
- A partial result with `partialReason` `interrupted` settles to the result document's `costUsd`.
  Runs that were in flight when the interrupt landed may have billed more than the document counts.
- A missing result whose stderr matches `LOAD_ERROR` anywhere (for example the text "invalid case"
  inside an unrelated message) settles to 0 USD, though the CLI may have run cases before it printed
  that text.
- The key-leak path (`sk-ant-` found in the result, a trace, or the output directory) returns before
  any settle, so the reservation stays at the cap. This overcounts rather than undercounts, and it
  leaves a reservation that only the `settle` command can close.

## Scope
1. Settle an interrupted partial result at no less than its reservation, or flag the row so
   `summarize` and `eval-spend` show it as a lower bound.
2. Match `LOAD_ERROR` only against the CLI's own load-failure line, and settle to 0 USD only when no
   cases ran.
3. Decide what the key-leak path settles, and print the `settle` command when it leaves the
   reservation open.
4. An unhandled exception from `cli.run` (an `OSError`, a SIGTERM) leaves a reservation open with no
   message. Only `KeyboardInterrupt` prints the settle command, so catch these too and print it.
