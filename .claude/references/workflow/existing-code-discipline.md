
# Existing Code Discipline

Rules for working within an existing codebase. Complements code-quality (smells/style) and refactoring-and-debt (refactoring cadence).

## Match Existing Patterns

- **Never introduce a new pattern** alongside an existing one without explicitly flagging the inconsistency
- If the codebase uses pattern A for X, use pattern A — even if you prefer pattern B
- If you believe a pattern should change, propose the migration as a separate task

## Understand Before Deleting

Code may be used in ways not visible through static analysis:
- Reflection, dynamic dispatch, string-based lookup
- External consumers (APIs, plugins, downstream repos)
- Build scripts, code generation, or test infrastructure
- Feature flags or environment-conditional paths

**If unsure whether code is used: ask, don't delete.**

## Separate Refactoring from Features

- Different commits minimum, different branches preferred
- Never mix behavior changes with structural changes — reviewers can't tell what's intentional
- Refactoring should be verifiable independently (tests still pass, behavior unchanged)
