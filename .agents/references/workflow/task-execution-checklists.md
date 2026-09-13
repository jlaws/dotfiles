# Task Execution Checklists

Reusable checklists for implementing, reviewing, and completing tasks. Referenced by batch-refactor, execute-plan, team-investigate, and other workflow commands.

## Per-Task Execution Flow

For each task in a plan, follow this cycle:

1. **Implement** — follow the Implementation Checklist below
2. **Spec compliance check** — use the Spec Compliance Checklist below
3. **If spec issues:** fix and re-check. Repeat until pass.
4. **Code quality check** — use the Code Quality Checklist below (only after spec compliance passes)
5. **If quality issues:** fix and re-check. Repeat until pass.
6. **Mark task complete**, move to next

---

## Implementation Checklist

Before starting a task, ensure you have:
- **Full task description** — requirements, acceptance criteria
- **Context** — where this fits, dependencies, architectural constraints
- **Constraints** — what NOT to change, scope boundaries

While implementing:
1. Implement exactly what the task specifies
2. Write tests (following TDD if required)
3. Verify implementation works
4. Commit your work

**If anything is unclear, ask before guessing.**

---

## Spec Compliance Checklist

Verify the implementation matches the specification — nothing more, nothing less.

**CRITICAL: Do not trust your own summary. Read the actual code.**

**Missing requirements:**
- Did I implement everything requested?
- Are there requirements I skipped or missed?
- Did I claim something works but didn't actually implement it?

**Extra/unneeded work:**
- Did I build things that weren't requested?
- Did I over-engineer or add unnecessary features?
- Did I add "nice to haves" not in spec?

**Misunderstandings:**
- Did I interpret requirements differently than intended?
- Did I solve the wrong problem?

**Verify by reading code, not by trusting your memory.**

**Result:** PASS or FAIL with specific issues (file:line references)

---

## Code Quality Checklist

**Only perform after spec compliance passes.**

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

**Result:** PASS / PASS WITH NOTES / NEEDS CHANGES

---

## Context Passing Template

When documenting work for handoff between tasks:

```
Context for next task:

Completed:
- {summary of work done}
- {key findings or decisions}

Remaining work:
- {specific tasks}
- {constraints}

Success criteria:
- {measurable outcomes}
```

---

## Red Flags — Stop and Reassess

- Skipping reviews (spec compliance OR code quality)
- Accepting "close enough" on spec compliance
- Starting quality review before spec review passes
- Moving to next task while reviews have open issues
- Scope creep beyond what was requested
- "While I'm here" additions not in the spec
- Tests passing but not actually testing the new behavior

## Common Task Mistakes

| Bad | Good |
|-----|------|
| "Fix all the tests" (too broad) | "Fix specific-test-file.test.ts" (focused) |
| "Fix the race condition" (no context) | Paste error messages and test names |
| No constraints | "Do NOT change production code" |
| "Fix it" (vague output) | "Return summary of root cause and changes" |
| No scope declared | "Files (read-write): src/auth/** — do not touch other paths" |
