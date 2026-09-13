
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

## Surface Hidden Assumptions

Watch for and document when you encounter:
- Implicit ordering dependencies (init before use, A before B)
- Undocumented invariants (field X is always non-null after method Y)
- Concurrency assumptions (single-threaded, lock held, queue ordering)
- Environment assumptions (only works on macOS, requires specific env vars)

## State Your Assumptions

State them numbered and falsifiable — something a reader can check and call wrong. "Inputs are under
10k rows and fit in memory" is an assumption. "The code should be maintainable" is not.

Cover whichever rows the work actually touches:

- **Data:** shape, volume, trust level, encoding, and what a malformed input looks like
- **Failure:** what happens on timeout, partial write, or a downstream 500 — retry, fail loud, or degrade
- **Boundaries:** who calls this, what is public API versus internal, what backwards-compat it owes
- **State:** concurrency, idempotency, transactionality, ordering guarantees
- **Environment:** runtime version, where it deploys, what it is allowed to reach
- **Scope:** what you are deliberately not doing, and what you are leaving as a TODO
- **Testing:** what you will cover, and what you will leave uncovered

Auth, money, migrations, and deletion get more suspicion than the rest — there, be more skeptical of
your own assumptions than the work seems to warrant.
