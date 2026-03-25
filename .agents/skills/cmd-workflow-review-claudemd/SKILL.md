---
name: cmd-workflow-review-claudemd
description: "Analyze recent conversation history to improve CLAUDE.md files — find violated instructions, missing patterns, and outdated rules. Use when tuning Claude Code behavior or after a batch of sessions. Do NOT use for quick questions (edit files directly instead)."
disable-model-invocation: true
---
# Review CLAUDE.md

Scope: the user's provided input

If no arguments provided, analyzes both global and local CLAUDE.md against last 20 conversations.

**Do NOT use subagents or parallel agents. Process all analysis linearly.**
