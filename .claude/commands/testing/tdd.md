---
name: tdd
description: "Autonomous TDD implementation loop — write tests first, run to confirm failure, implement until green, lint, commit. Use when adding a feature or fixing a bug with test-driven development. Do NOT use for debugging existing failures (use /debug) or large multi-file refactors (use /batch-refactor)."
argument-hint: "<feature or bug description>"
---

Load and follow the `testing/test-driven-development` skill for methodology.
Load the `workflow/code-quality` skill for formatting and lint standards.

Feature/bug: $ARGUMENTS

If no arguments provided, ask what the user wants to implement.

## Execution Loop

Follow this strict sequence — do not skip or reorder steps:

1. **Explore**: Read existing code around the feature area. Identify edge cases, existing patterns, and test infrastructure (test runner, fixtures, helpers).

2. **Write tests**: Write comprehensive tests covering happy path, error cases, and edge cases. These tests MUST fail initially — if any pass, you're testing existing behavior (fix the test).

3. **Verify RED**: Run the full test suite. Confirm every new test fails for the expected reason (feature missing, not typos or import errors). Paste output.

4. **Implement**: Make each test pass one-by-one. After each change, run the full test suite. Write minimal code — don't add features beyond what the tests require.

5. **Lint**: Run the project's formatter and linter (check Makefile, package.json scripts, or pyproject.toml for commands). Fix all issues.

6. **Final verification**: Run the full test suite one last time. Paste output confirming all tests pass and output is clean.

7. **Commit**: Create a commit summarizing what was implemented and test coverage achieved.

Do not move to step 4 until ALL tests from step 2 are written.
Do not skip running tests between implementations in step 4.
