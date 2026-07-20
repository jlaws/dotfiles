---
name: j-config-audit
description: "Security audit of Claude, Codex, Gemini, and shared agent configuration for secrets, over-broad permissions, and prompt-injection vectors."
argument-hint: "<scope: secrets|permissions|injection|path>"
---

Load the `config-security-audit` skill by reading `~/.agents/skills/config-security-audit/SKILL.md` before doing anything else. Run the full config-security scan.

Scope: $ARGUMENTS

If no argument provided, scan all configuration trees.
