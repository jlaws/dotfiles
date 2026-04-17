---
name: output-completeness
description: "Enforce complete output generation — no truncation, no stubs, no skeleton code. Load for large code implementations, multi-part responses, complex refactors, full documentation sections, or any task where truncation risk is high. Distinct from verification-before-completion (which governs claims about work state)."
compatibility: claude-code
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Output Completeness

**Core principle:** A partial output is a broken output. Do not optimize for brevity — optimize for completeness.

**Violating the letter of this rule is violating the spirit of this rule.**

## The Iron Law

```
NEVER TRUNCATE. NEVER STUB. NEVER DEFER.
If you started it, finish it.
```

## Banned Code Patterns

These patterns are **never acceptable** in generated code:

| Pattern | Example | Status |
|---------|---------|--------|
| Ellipsis placeholder | `// ...` | BANNED |
| TODO stub | `// TODO: implement this` | BANNED |
| Comment placeholder | `/* rest of implementation */` | BANNED |
| Skeleton function | `func foo() { pass }` | BANNED |
| Deferred implementation | `// implement X here` | BANNED |
| Partial class | Class with unimplemented methods | BANNED |

## Banned Prose Patterns

These patterns are **never acceptable** in generated text:

| Pattern | Example | Status |
|---------|---------|--------|
| Backward reference | "as mentioned above" | BANNED |
| Ellipsis substitution | "... (similar to above)" | BANNED |
| Section stub | "### Section Title\n\n(Content TBD)" | BANNED |
| Exercise deferral | "I'll leave X as an exercise" | BANNED |
| Implicit truncation | Ending a response mid-list or mid-section | BANNED |

## Required Process

```
1. SCOPE: Read the full request. Identify every deliverable.
2. PLAN: List all sections/functions/files to generate.
3. GENERATE: Produce each item completely before moving to the next.
4. CROSS-CHECK: Re-read the original request. List every deliverable.
         Confirm each one is present and complete.
5. ONLY THEN: Respond.
```

## Token Management

When a response genuinely cannot fit in one output:

**DO:**
```
[PAUSING — remaining sections: X, Y, Z. Reply "continue" to proceed.]
```

**DO NOT:**
- Silently truncate
- Use `// ...` or similar placeholders
- Emit the pause notice mid-sentence or mid-function

The pause notice must appear at a clean breakpoint (end of a function, end of a section).

## Distinction From Verification-Before-Completion

| Skill | Governs |
|-------|---------|
| `output-completeness` | Whether generated output is complete and untruncated |
| `verification-before-completion` | Whether claims about work state are backed by evidence |

Both must be applied for large implementations. They are complementary, not redundant.

## When To Load This Skill

Load proactively when:
- Implementing a complete feature (>1 file or >50 lines)
- Writing full documentation sections (not summaries)
- Executing a multi-part refactor
- Writing a research analysis or technical report
- Generating test suites (skeleton tests = broken tests)
- Any task where "partially done" would block the user

## Cross-References

- **workflow:llm-output-completeness** — root cause research and parameter tuning for truncation
- **workflow:completeness-principle** — project-level completeness standards
