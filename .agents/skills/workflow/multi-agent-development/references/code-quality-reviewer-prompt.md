# Code Quality Reviewer Prompt Template

Use this template when dispatching a code quality reviewer subagent.

**Purpose:** Verify implementation is well-built — clean, tested, maintainable.

**Only dispatch after spec compliance review passes.**

```
Agent:
  description: "Review code quality for Task N"
  prompt: |
    You are reviewing code quality for a recently implemented task.

    ## What Was Implemented

    [From implementer's report — summary of changes]

    ## Requirements Context

    [Task N from plan — so you understand what was being built]

    ## Diff to Review

    Base: [commit SHA before task]
    Head: [current commit SHA]

    Run: git diff <base>..<head>

    ## Your Job

    Review the implementation for:

    **Code Quality:**
    - Is the code clean, readable, and well-organized?
    - Do names accurately describe what things do?
    - Is there unnecessary complexity or duplication?
    - Does it follow existing codebase patterns and conventions?

    **Testing:**
    - Do tests actually verify behavior (not just coverage)?
    - Are edge cases tested?
    - Are tests maintainable and clear?
    - Do tests follow the project's testing patterns?

    **Architecture:**
    - Does the implementation fit the existing architecture?
    - Are abstractions at the right level?
    - Is there appropriate separation of concerns?

    **Potential Issues:**
    - Race conditions, error handling gaps
    - Security concerns (injection, auth, data exposure)
    - Performance issues (N+1 queries, unnecessary allocations)

    ## Report Format

    **Strengths:** What was done well

    **Issues:** (grouped by severity)
    - Critical: Must fix before merge
    - Important: Should fix, significant impact
    - Minor: Nice to have, low impact

    **Assessment:** PASS / PASS WITH NOTES / NEEDS CHANGES
```
