---
name: architecture-team-design
description: "System design suite — produce architecture documents from multiple specialist perspectives. Use when designing a new system or documenting existing architecture. Do NOT use for quick design questions (use /arch instead)."
argument-hint: "<directory-path> <system-description>"
---

Parse arguments: `$ARGUMENTS` must contain `<directory_path>` followed by `<description>`.
- First token = directory path, remainder = system description.
- If either is missing, ask the user.

**Do NOT use subagents or parallel agents. Process all design perspectives linearly.**

Conduct system design covering multiple specialist perspectives.

