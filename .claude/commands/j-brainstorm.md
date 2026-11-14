---
name: j-brainstorm
description: "Design-before-implementation workflow — explore intent, requirements, and design before writing code. Use when creating features, building components, or modifying behavior. Do NOT use after implementation is started (use /j-execute-plan instead)."
argument-hint: "<topic or feature description>"
---

Invoke the `design-first` skill via the Skill tool before doing anything else. Use it to explore intent, requirements, and design for the topic below. Load skill `analysis-output-patterns` for output structure.

In design-first's Phase 1, also read the project's conventions: the root instruction file (CLAUDE.md / AGENTS.md / GEMINI.md) and every doc it names for coding conventions, principles, or architecture rules, never filtered by topic keywords. Keep them as an ID-cited conventions digest, and make conformance to the conventions digest part of the design's self-review.

For scoping, product framing, or architecture options, you may delegate to a specialist agent via the Agent tool — `scope-reviewer` or `product-manager` (load `references/product/`) or `architecture-specialist` (loads `references/architecture/`). Verify their output before presenting.

Topic: $ARGUMENTS

If no arguments provided, ask what the user wants to build or design.
