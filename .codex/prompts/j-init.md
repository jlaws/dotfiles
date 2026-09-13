---
name: j-init
description: "Scaffold a repository for Claude, Codex, and Gemini -- agent config, project settings, toolchain, and a seed ADR, derived from what the repo already has. Use when starting a new project or when a repo has no agent config. Do NOT use for editing config in an already-scaffolded repo (edit the files directly)."
argument-hint: "<path to repository, or a description of the project to create>"
---

Invoke the `project-scaffolding` skill before doing anything else.

Target: $ARGUMENTS

If no arguments provided, scaffold the current working directory.

Read the repository before asking anything. If it has a README or code, infer purpose and stack and
ask the user to correct you. If it is empty, invoke the `design-first` skill to establish purpose,
requirements, and constraints first.

For stack idioms and project layout you may delegate to `language-specialist` (loads
`references/languages/`). Verify its output, and never take a version number from it: version
currency comes from a registry query.
