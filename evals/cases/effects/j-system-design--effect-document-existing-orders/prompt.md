---
tags: [effects, effect, heavy]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "document existing code and flag the command's red flags, such as circular dependencies, mixed responsibilities, missing error handling, and exposed sensitive data, without proposing unrelated features; .claude/commands/j-system-design.md; hard because the circular import and the card data in the audit log only show when reading across files"
---

/kb:j-system-design document the architecture of the existing order service in ./src, and review the design critically. Save it to ./architecture.md with a component catalog, a dependency description, risks and open questions, and an implementation priority order for fixing what you find. Name every structural problem you find with the file paths involved, and do not propose new features or components beyond what fixing those problems needs.
