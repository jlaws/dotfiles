---
name: skill-audit
description: "Audit the .claude/ knowledge base — skills, commands, agents, references, config, and cross-references for conformance and integrity. Use when creating or modifying any .claude/ asset to validate compliance."
argument-hint: "<scope: skills|commands|agents|references|config|path>"
---

Load and follow the `workflow/skill-audit` skill to audit the `.claude/` knowledge base.

Scope: $ARGUMENTS

If no arguments provided, audit all asset types. Otherwise scope to the specified type or path (e.g., `skills`, `agents`, `commands/workflow/ml`, `references/security`).
