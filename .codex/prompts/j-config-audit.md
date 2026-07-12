---
name: j-config-audit
description: "Security audit of the agent-config surface across all tool trees (.claude, .codex, .cursor, .gemini, .agents) — leaked secrets, over-broad tool permissions, and prompt-injection vectors. Use before publishing or syncing configs, or in CI. Do NOT use for application code security (use /j-audit) or KB conformance (use /j-skill-audit)."
argument-hint: "<scope: secrets|permissions|injection|path>"
---

Load the `config-security-audit` skill by reading `~/.agents/skills/config-security-audit/SKILL.md` before doing anything else. Run the full config-security scan.

Scope: $ARGUMENTS

If no argument provided, scan all configuration trees.
