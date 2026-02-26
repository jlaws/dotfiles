---
name: writing-plans
description: "Structured methodology for writing implementation plans with bite-sized tasks, exact file paths, and TDD integration. Use when you have a spec or requirements for a multi-step task and need to create a detailed plan before writing code. Do NOT use for simple single-file changes or bug fixes."
compatibility: claude-code
allowed-tools: Read, Grep, Glob, Bash
---

# Writing Plans

## Overview

Write comprehensive implementation plans assuming the engineer has zero codebase context. Document everything they need: which files to touch, complete code samples, how to test, exact commands with expected output. Bite-sized tasks. DRY. YAGNI. TDD. Frequent commits.

Target audience: skilled developer unfamiliar with your codebase and toolset.

**Save plans to:** `docs/plans/YYYY-MM-DD-<feature-name>.md`

## Plan Document Header

Every plan MUST start with:

```markdown
# [Feature Name] Implementation Plan

**Goal:** [One sentence describing what this builds]

**Architecture:** [2-3 sentences about approach]

**Tech Stack:** [Key technologies/libraries]

---
```

## Bite-Sized Task Granularity

Each step is one action (2-5 minutes):

- "Write the failing test" — step
- "Run it to make sure it fails" — step
- "Implement the minimal code to make the test pass" — step
- "Run the tests and make sure they pass" — step
- "Commit" — step

If a step takes more than 5 minutes, split it further.

## Task Structure

````markdown
### Task N: [Component Name]

**Files:**
- Create: `exact/path/to/file.ext`
- Modify: `exact/path/to/existing.ext:123-145`
- Test: `tests/exact/path/to/test.ext`

**Step 1: Write the failing test**

```language
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

**Step 2: Run test to verify it fails**

Run: `test-command path/to/test::test_name`
Expected: FAIL with "function not defined"

**Step 3: Write minimal implementation**

```language
def function(input):
    return expected
```

**Step 4: Run test to verify it passes**

Run: `test-command path/to/test::test_name`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/path/test.ext src/path/file.ext
git commit -m "feat: add specific feature"
```
````

## Requirements

- **Exact file paths** — always, no "add to the appropriate file"
- **Complete code** — paste actual code, not "add validation here"
- **Exact commands** — with expected output, not "run the tests"
- **TDD integration** — each task = failing test → verify fail → implement → verify pass → commit
- **Self-contained tasks** — each task can be understood and executed independently
- **Frequent commits** — one commit per task or logical unit

## Execution Handoff

After saving the plan, present execution options:

```
Plan saved to `docs/plans/<filename>.md`. Execution options:

1. **Execute now** — I'll work through tasks in batches with review checkpoints
   (uses workflow/executing-plans skill)

2. **Execute in new session** — Open new session and run /execute-plan
   (better for large plans — fresh context per batch)

3. **Manual** — You execute the plan yourself

Which approach?
```

## Common Mistakes

| Bad | Good |
|-----|------|
| "Add validation" | Complete validation code with specific checks |
| "Update the tests" | Exact test code with expected assertions |
| "Run the test suite" | `npm test -- --grep "feature"` → Expected: 5 pass |
| "Modify the config" | Exact config changes with file path and line numbers |
| Tasks that take 30+ min | Split into 2-5 minute steps |
| Assuming reader knows the codebase | Explain where things are and why |
