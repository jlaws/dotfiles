---
name: j-skill-audit
description: "Audit the .claude/ knowledge base — skills, commands, agents, references, config, cross-references, and documentation currency for conformance and integrity. Use when creating or modifying any .claude/ asset to validate compliance. Do NOT use for quick checks (inspect files directly instead)."
argument-hint: "<scope: skills|commands|agents|references|config|path>"
model: sonnet
---

Invoke the `skill-audit` skill via the Skill tool before doing anything else. Run the full conformance audit.

Scope: $ARGUMENTS

If no argument provided, audit all asset types.
