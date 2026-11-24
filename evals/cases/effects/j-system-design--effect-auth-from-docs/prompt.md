---
tags: [effects, effect, heavy]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "design from the relevant repo documents only and set unrelated ones aside, record the key decision as an ADR with alternatives and trade-offs, and plan risk-first in vertical slices; .claude/commands/j-system-design.md; hard because the billing document sits in the same docs folder and invites itself into the design"
---

/kb:j-system-design design the authentication system for our internal portal, using whatever requirements the repo's docs give for it. Save the architecture document to ./architecture.md with a component catalog, the key design decisions with alternatives considered, risks, and a risk-first implementation plan made of vertical slices. Record the session-token decision as an ADR at ./docs/adr/auth/session-tokens.md. Use only documents that relate to authentication, and say in architecture.md which documents you used and which you set aside.
