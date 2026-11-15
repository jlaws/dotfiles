---
type: llm
focus: { source: file, path: plans/plan.md }
---

All claims are required.
PASS if the plan opens with a purpose statement describing user-visible behavior (for example that `python3 -m calc median ...` prints the median), not a task verb like 'implement median'.
PASS if the first phase fetches and branches off origin/main and writes the README section before any test or code task.
PASS if each code phase puts a failing test and a run showing it fails before the implementation step, with a run command and expected result.
PASS if the plan ends with a validation phase that names a gate command and expected output, or says N/A with a stated reason, and states PR boundaries with estimated changed lines.
FAIL if the plan's code blocks contain complete implementations of median or the command line entry point instead of signatures and test assertions.
