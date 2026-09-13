---
name: j-init
description: "Scaffold a repository for Claude, Codex, and Gemini -- agent config, project settings, toolchain, and supporting docs, derived from what the repo already has. Use when starting a new project, when a repo has no agent config, or when its config is stale or partial. Do NOT use for creating knowledge-base assets such as commands or skills (use /j-new)."
---

# Init

Invoke the `project-scaffolding` skill before doing anything else. It owns the method: what to read
before asking, what to interview for, where version numbers come from, and what gets written.

For stack idioms and project layout you may delegate to `language-specialist` (loads
`references/languages/`). Verify its output, and never take a version number from it.

Target: $ARGUMENTS

If no arguments provided, scaffold the current working directory.
