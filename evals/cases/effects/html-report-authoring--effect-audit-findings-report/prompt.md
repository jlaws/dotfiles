---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "a self-contained report page with TOC jump links, severity-coded findings, both color schemes, and an inline SVG with role img and aria-labelledby on its title and desc; .claude/skills/html-report-authoring/SKILL.md; hard because a capable model labels the diagram with aria-label or a caption and skips the title-and-desc wiring the skill requires"
---

Build a report of these security audit findings as one HTML file at ./report.html. It must be a single self-contained file that works when opened offline, easy to scan, readable in light and dark mode, and include one inline diagram of the request path with an accessible text description for screen readers.

Findings:
- F1 critical: session token logged in plaintext by auth-service (logger.go:88)
- F2 high: admin endpoint /admin/export has no rate limit (routes.go:41)
- F3 medium: password reset links valid for 72 hours (reset.go:19)
- F4 low: verbose stack traces returned on 500 responses (errors.go:7)

Request path for the diagram: client, then gateway, then auth-service, then orders-api, then database.
