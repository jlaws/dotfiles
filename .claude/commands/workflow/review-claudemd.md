---
name: review-claudemd
description: "Analyze recent conversation history to improve CLAUDE.md files — find violated instructions, missing patterns, and outdated rules. Use when tuning Claude Code behavior or after a batch of sessions. Do NOT use for quick questions (edit files directly instead)."
argument-hint: "<scope: global|local|number|empty>"
---

Load and follow the `workflow/multi-agent-development` skill with CLAUDE.md review agents to analyze conversation history:

Scope: $ARGUMENTS

If no arguments provided, analyzes both global and local CLAUDE.md against last 20 conversations.
