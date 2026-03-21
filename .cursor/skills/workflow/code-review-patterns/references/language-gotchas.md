# Language-Specific Review Gotchas

Quick-reference of common language-specific issues to watch for during code review.

## Python
- Mutable default arguments (`def fn(items=[])` -- use `None`)
- Bare `except:` catching everything
- Mutable class attributes shared across instances
- Late binding closures in loops
- Iterator exhaustion (consuming a generator twice)

## TypeScript / JavaScript
- `any` type defeating type safety (use `unknown`)
- Unhandled async errors (missing try/catch on await)
- Prop mutation in React components
- `==` vs `===` coercion bugs
- Prototype pollution

## Go
- Goroutine leaks (missing context cancellation)
- Unchecked errors (`err` ignored)
- Nil pointer dereference
- `defer` in loops (resource accumulation)

## Bash
- Unquoted variables causing word splitting
- Missing `set -euo pipefail`
- Using `[ ]` instead of `[[ ]]`

## Swift
- Retain cycles (missing `[weak self]`)
- Force unwrapping (`!`) without safety check
- Main thread violations for UI updates

## Rust
- Unnecessary `clone()` defeating borrow checker
- `unsafe` blocks without justification
- Lifetime issues from overly complex borrowing
