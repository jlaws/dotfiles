---
description: "Multi-agent team code review — parallel specialist reviewers for security, quality, testing, and language-specific analysis."
---

Before invoking the skill, perform pre-flight checks:

1. **Verify branch**: Confirm current branch is NOT main/master — fail fast if so.
2. **Check commits ahead**: `git log main..HEAD --oneline` — fail fast if no commits ahead.
3. **Get full diff**: `git diff main...HEAD`
4. **Detect languages** from changed file extensions.
5. **Get changed files list**: `git diff main...HEAD --name-only`

Then orchestrate the review team:

1. **Create a team** with TeamCreate.
2. **Enter delegate mode** (Shift+Tab) — lead coordinates and synthesizes only, does not edit files.
3. **Create tasks** with TaskCreate — one per review perspective, each declaring:
   - Review focus and skill to load
   - File ownership: **shared-read-only for all reviewers** (no edits)
   - The full diff and changed files list as context
4. **Spawn specialist reviewers** (Explore type, read-only), assign tasks via TaskUpdate:
   - **security-reviewer**: Loads security:security-analysis perspective. Reviews for STRIDE threats, vulnerability patterns, secrets, injection vectors.
   - **quality-reviewer**: Loads workflow:code-quality + workflow:code-review-excellence perspectives. Reviews for code smells, edge cases, error handling, naming, DRY.
   - **test-reviewer**: Loads testing:language-testing-patterns perspective. Identifies coverage gaps, missing edge case tests, test quality issues.
   - **language-reviewer**: Loads the detected languages:*-patterns skill. Reviews for language-specific gotchas and idiom violations.
5. **Collect findings** from all reviewers via TaskList, deduplicate, and merge into a severity-ranked report.
6. **Shut down team** after all findings are collected.

Invoke the workflow:diff-review skill output format for the final report structure: $ARGUMENTS
