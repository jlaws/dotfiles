
# Existing Code Discipline

Rules for working within an existing codebase. Complements code-quality (smells/style) and refactoring-and-debt (refactoring cadence).

## Read Before Modifying

- **Read the entire file** before changing any part of it — not just the section you plan to edit
- Understand the file's structure, conventions, and how your target section relates to the rest
- Check for file-level comments, configuration blocks, or initialization that affects your change

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

## Scope Guard

- If a "small fix" grows mid-implementation — **STOP and ask**
- Define your change boundary before starting; resist scope creep
- A fix that touches 3 files should not silently become a fix that touches 12

## Separate Refactoring from Features

- Different commits minimum, different branches preferred
- Never mix behavior changes with structural changes — reviewers can't tell what's intentional
- Refactoring should be verifiable independently (tests still pass, behavior unchanged)

## Surface Hidden Assumptions

Watch for and document when you encounter:
- Implicit ordering dependencies (init before use, A before B)
- Undocumented invariants (field X is always non-null after method Y)
- Concurrency assumptions (single-threaded, lock held, queue ordering)
- Environment assumptions (only works on macOS, requires specific env vars)
