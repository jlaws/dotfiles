---
name: j-brainstorm
description: "Design-before-implementation workflow — explore intent, requirements, and design before writing code. Use when creating features, building components, or modifying behavior. Do NOT use after implementation is started (use $cmd-j-execute-plan instead)."
argument-hint: "<topic or feature description>"
---

Invoke the `design-first` skill before doing anything else. Use it to explore intent, requirements, and design for the topic below.

For scoping, product framing, or architecture options, you may delegate to a specialist agent — `scope-reviewer` or `product-manager` (load `.agents/references/product/`) or `architecture-specialist` (loads `.agents/references/architecture/`). Verify their output before presenting.

Topic: $ARGUMENTS

If no arguments provided, ask what the user wants to build or design.
