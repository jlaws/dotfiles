---
name: team-design
description: "Multi-agent system design suite — parallel specialist agents produce architecture documents. Use when designing a new system or documenting existing architecture. Do NOT use for quick design questions (use /arch instead)."
argument-hint: "<directory-path> <system-description>"
---

Load and follow the `workflow/multi-agent-development` skill to conduct parallel system design.

Parse arguments: `$ARGUMENTS` must contain `<directory_path>` followed by `<description>`.
- First token = directory path, remainder = system description.
- If either is missing, ask the user.
