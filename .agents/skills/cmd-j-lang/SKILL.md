---
name: cmd-j-lang
description: "Language patterns consultation — idioms, tooling, and project scaffolding. Use when setting up a project, choosing patterns, or working in Python, JS/TS, Go, Rust, Swift, or other languages. Do NOT use for framework-specific help (use /j-frontend or domain specialist instead)."
disable-model-invocation: true
---

# Language Patterns Consultation

Before starting, gather diagnostic context:

1. **Detect primary language** from config files (package.json, pyproject.toml, Cargo.toml, go.mod, Package.swift, Gemfile).
2. **Identify tooling** by searching for linter configs, formatter configs, build tool settings.
3. **Check project conventions** for existing patterns, module structure, and coding standards.
4. **Get scope overview** of the target area (if the user's provided input specifies a topic, scope to that; otherwise detect language from the project root).

For deep language guidance, delegate to the `language-specialist` agent, passing the detected language/tooling and the request. It loads its skills (language-testing-patterns, test-driven-development) and the `.agents/references/languages/` library, then returns idiomatic guidance. Verify its output before presenting.

Help with: the user's provided input
