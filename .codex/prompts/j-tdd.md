---
name: j-tdd
description: "Autonomous TDD implementation loop — write tests first, run to confirm failure, implement until green, lint, commit. Use when adding a feature or fixing a bug with test-driven development. Do NOT use for debugging existing failures (use $cmd-j-debug)."
argument-hint: "<feature or bug description>"
---

Feature/bug: $ARGUMENTS

If no arguments provided, ask what the user wants to implement.

Invoke the `test-driven-development` skill before coding. Then follow this execution loop in order — do not skip or reorder steps:

1. **Explore** — Read existing code around the feature area. Identify edge cases, existing patterns, and test infrastructure (test runner, fixtures, helpers).
2. **Write tests** — Cover happy path, error cases, and edge cases. Tests MUST fail initially.
3. **Verify RED** — Run the full test suite. Confirm every new test fails for the expected reason (feature missing, not typos or import errors). Paste output.
4. **Implement** — Make each test pass one-by-one. After each change, run the full test suite. Write minimal code.
5. **Lint** — Run the project's formatter and linter (check Makefile, package.json scripts, or pyproject.toml for commands). Fix all issues.
6. **Final verification** — Run the full test suite. Paste output confirming all tests pass and output is clean.
7. **Validate docs** — Apply the `documentation-validation` gate: if the feature/fix changed public surface or documented behavior, update README/API/CHANGELOG (or declare N/A with a reason).
8. **Commit** — Create a commit summarizing what was implemented and test coverage achieved.

Do not move to step 4 until ALL tests from step 2 are written. Do not skip running tests between implementations in step 4.

For a large feature, you may delegate test authoring (steps 2-3) to the `test-writer` agent — it loads test-driven-development + language-testing-patterns + `.agents/references/testing/`. Always verify RED/GREEN yourself; a subagent's "tests pass" is not proof.
