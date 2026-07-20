---
name: j-defrag
description: "Scan the codebase for fragmentation — duplicated components, inconsistent patterns, files to consolidate, logic to share — and fix it in verified batches. Use to keep the codebase coherent on a regular cadence. Do NOT use for single-file cleanups (edit directly) or diff review before merge (use /j-diff-review)."
argument-hint: "<optional path or subsystem to scope the scan; default whole repo>"
model: opus
---

Load skills `refactoring-and-debt` and `code-quality` via the Skill tool before doing anything else. Defragment the codebase: scan for fragmentation and fix it in verified batches so the next thing built lands in a more coherent codebase.

## Phase 1 — Scope

Resolve the scan scope: if arguments are provided, limit to that path or subsystem: $ARGUMENTS. Else scan the whole repo.

Identify the project's test command (from CLAUDE.md, `package.json`, `Makefile`, `pyproject.toml`, etc.) — verified batches need it. If no test command exists, warn and fall back to report-only (Phases 2-3, skip fixing).

## Phase 2 — Scan for fragmentation

Detect and group four fragmentation classes (use Grep/Glob to find candidates; lean on `code-quality` smell detection and `refactoring-and-debt`'s smell→refactoring map):

- **Duplicated components / logic** — near-identical blocks, copy-paste, repeated rules → extract/share.
- **Inconsistent patterns** — the same job done N different ways; naming/structure drift → converge on one.
- **Files that should be consolidated** — scattered related code, over-split modules → merge.
- **Logic that should be shared** — a local reimplementation of an existing util → reuse the util.

Group findings into ranked clusters, ordered by value (impact vs risk).

## Phase 3 — Present plan

Output the ranked clusters as a table: class, files, proposed fix, risk. This is the flag step — kept, but not the endpoint.

## Phase 4 — Fix in verified batches

For each cluster, highest value first:
1. Confirm tests green (baseline).
2. Apply the consolidation/extraction/dedup using `refactoring-and-debt` safe sequences.
3. Re-run tests; they must stay green (`verification-before-completion`).
4. One atomic commit per cluster (imperative subject, <72 chars).
5. If a cluster renamed or moved public surface (exported names, CLI flags, config keys, or KB asset names/paths), apply the `documentation-validation` gate before the commit — update docs and any required native or mirror copies, or declare N/A.

Stop when: tests fail and cannot be fixed in <=2 attempts, or two consecutive rounds make no progress. Then report and hand back.

## Phase 5 — Summary

Report clusters fixed, clusters deferred (with reason), and residual fragmentation to tackle on the next pass. Defrag is a recurring cadence, not a one-shot — each pass leaves the codebase more coherent for the next.

### Cross-References

- **skill:refactoring-and-debt** — fix mechanics, smell→refactoring map, safe refactoring sequences.
- **skill:code-quality** — DRY and code-smell detection for the scan phase.
- **skill:verification-before-completion** — the green-tests gate between batches.
- **/j-diff-review** — use instead when reviewing a single diff before merge.
- **skill:documentation-validation** — doc gate when a cluster renames or moves public surface.
