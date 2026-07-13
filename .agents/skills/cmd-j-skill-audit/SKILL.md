---
name: cmd-j-skill-audit
description: "Audit the .claude/ knowledge base — skills, commands, agents, references, config, cross-references, and documentation currency for conformance and integrity. Use when creating or modifying any .claude/ asset to validate compliance. Do NOT use for quick checks (inspect files directly instead)."
disable-model-invocation: true
---

# Knowledge Base Audit

Invoke the `skill-audit` skill before doing anything else. Run the full conformance audit.

Scope: the user's provided input

If no argument provided, audit all asset types.
