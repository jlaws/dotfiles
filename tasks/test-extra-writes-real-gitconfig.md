# Task: Stop `tests/test_extra.py` writing the real `~/.gitconfig`

## Context
Found on 2026-11-24 while reviewing the eval case audit: `~/.gitconfig` changed its modification
time during every test run.

## Problem
`.extra` runs `git config --global user.name` and `git config --global user.email` each time it is
sourced. The four ulimit tests at the top of `tests/test_extra.py` source `.extra` with the real
`HOME`, so every `make test` rewrites the user's `~/.gitconfig`. The values are the ones `.extra`
already sets in every new shell, so the content does not change, but a test suite should not write
outside its temp directories. The second class in the same file already runs under a temp `HOME`.

## Scope
1. Run the ulimit tests with `HOME` pointed at a temp directory, as `_zsh` does.
2. Confirm with `stat -f '%m' ~/.gitconfig` before and after `make test`.
