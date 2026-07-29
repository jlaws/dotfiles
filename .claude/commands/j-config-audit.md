---
name: j-config-audit
description: "Security audit of Claude, Codex, Gemini, and shared agent configuration for secrets, over-broad permissions, and prompt-injection vectors. Use when reviewing agent config after changing settings, hooks, or permissions. Do NOT use for application code security (use /j-audit)."
argument-hint: "<scope: secrets|permissions|injection|path>"
model: sonnet
effort: xhigh
---

Invoke the `config-security-audit` skill via the Skill tool before doing anything else. Run the full config-security scan.

Scope: $ARGUMENTS

If no argument provided, scan all configuration trees.
