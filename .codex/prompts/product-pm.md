---
name: product-pm
description: "Product management consultation — PRDs, roadmaps, opportunity assessment, and launch planning. Use when evaluating what to build, writing requirements, or planning launches. Do NOT use for technical architecture (use /arch instead) or business KPIs (use /biz instead)."
argument-hint: "<question-or-task>"
---

Before invoking the subagent, gather diagnostic context:

1. **Detect project type** from package.json, pyproject.toml, Cargo.toml, or similar — understand what the product is.
2. **Check for existing product artifacts** by searching for PRD, requirements, roadmap, or spec files (docs/, specs/, .md files with "requirement" or "prd" in the name).
3. **Check analytics setup** by searching for analytics, tracking, or event instrumentation code (e.g., track(), analytics., mixpanel, amplitude, segment).
4. **Find user research artifacts** by searching for user interviews, feedback, surveys, or persona files.
5. **Get scope overview** of the target area (if $ARGUMENTS specifies a feature or component, scope to that).

Use the product-manager subagent to help with: $ARGUMENTS
