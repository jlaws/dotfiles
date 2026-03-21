---
name: cmd-product-scope-review
description: "Strategic scope review — challenge assumptions about WHAT to build before design work begins. Four modes: EXPAND, SELECTIVE EXPAND, HOLD, REDUCE. Use when evaluating feature scope, validating MVP boundaries, or before /cmd-workflow-brainstorm. Do NOT use for technical design (use /cmd-architecture-arch) or after implementation starts."
disable-model-invocation: true
---

# Scope Review

Before applying scope review methodology, gather context:

1. **Detect project type** from package.json, pyproject.toml, Cargo.toml, or similar.
2. **Find existing specs** by searching for PRD, requirements, roadmap, or spec files.
3. **Check recent history** — `git log --oneline -20` for related work on the topic.
4. **Identify constraints** — deadlines, dependencies, or resource limits mentioned in docs.

Read `.cursor/references/product/scope-review-methodology.md` for the full framework (if available).

Apply all four scope modes (EXPAND, SELECTIVE EXPAND, HOLD, REDUCE) to the topic specified by the user.

For each mode, generate:
- What the scope would look like under this mode
- Key assumptions being made
- Risks and trade-offs

Conclude with:
- **Recommended mode** with reasoning
- **Key assumptions** that should be validated
- **Next step**: typically `/cmd-workflow-brainstorm` (HOW to build) or `/cmd-workflow-write-plan` (if scope is clear)
