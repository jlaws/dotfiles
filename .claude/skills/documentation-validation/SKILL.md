---
name: documentation-validation
description: "Use when validating docs before shipping a change."
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Documentation Validation

**A change is not done until its docs match it.**

This is the lightweight, per-change gate. For the heavy post-release sweep over a
whole diff range, use `post-ship-doc-sync`.

## When To Apply

Run this gate before claiming any change-shipping task complete, before committing,
and before opening or updating a PR — for every feature, behavior change,
refactor/rename, or new KB asset.

## The Gate

```
BEFORE claiming a change-shipping task done:
1. CLASSIFY the change (table below).
2. CHECK the docs that row points to — read them, compare against the diff.
3. RESOLVE: update the doc in the same change, OR declare N/A with a one-line reason.
4. ONLY THEN claim done.
```

An unstated doc decision is a failed gate. Silence is not N/A. Stale docs block
completion the same way failing tests do.

## Change-Type → Docs Matrix

Use `post-ship-doc-sync`'s Step 3 staleness heuristic (API ref, setup/install,
architecture, config ref, CLI usage, feature docs) to decide *whether* a doc is
stale. This table decides *which* docs to open per change.

| Change type | Product docs to check | KB self-docs to check |
|-------------|-----------------------|-----------------------|
| Add feature | README feature list, API reference, CHANGELOG, usage/quickstart | (usually N/A unless a KB asset is added) |
| Modify behavior | API reference, config/CLI reference, CHANGELOG, any doc describing the old behavior | — |
| Refactor / rename | Public names in README/API docs, import paths, config keys, CLI flags | cross-references that name the renamed asset |
| Add new KB asset | — | `description` frontmatter + body, KB-structure docs, MEMORY.md index (if present), cross-references, native command/agent parity, and shared workflow/reference mirror parity |

Product docs = README / API / CHANGELOG / usage. KB self-docs = the asset's own
`description` and body, the KB-structure index, native asset parity, and any
required shared workflow or reference mirror.

## Declaring N/A

If a change genuinely needs no doc update, state it explicitly:

`Docs: N/A — internal refactor, no public surface or documented behavior changed.`

## Red Flags

- Claiming done without a stated doc decision (update or explicit N/A)
- "Docs can follow in a separate PR" / "I'll update docs later"
- Renaming a public symbol but leaving its old name in README/API docs
- Changing behavior described in a doc without touching that doc
- Adding a KB asset without registering it or satisfying its native/mirror parity rule
- Treating this as the full post-release sweep (that is `post-ship-doc-sync`)

## Cross-References

- **post-ship-doc-sync** — the heavy post-release sweep over a whole diff range; source of the Step 3 staleness heuristic this gate reuses. This skill is its lightweight per-change counterpart.
- **verification-before-completion** — verdict grammar and evidence hierarchy for reporting what you checked. "Docs current" means updated, or an explicit N/A with a reason.
