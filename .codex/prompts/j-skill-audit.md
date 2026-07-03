---
name: j-skill-audit
description: "Audit the .claude/ knowledge base — skills, commands, agents, references, config, and cross-references for conformance and integrity. Use when creating or modifying any .claude/ asset to validate compliance. Do NOT use for quick checks (inspect files directly instead)."
argument-hint: "<scope: skills|commands|agents|references|config|path>"
---

Invoke the `skill-audit` skill before doing anything else. Run the full conformance audit.

Scope: $ARGUMENTS

If no argument provided, audit all asset types.
