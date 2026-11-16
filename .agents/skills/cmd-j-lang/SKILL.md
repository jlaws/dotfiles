---
name: cmd-j-lang
description: "Use when invoking the j-lang workflow."
---

# Language Patterns Consultation

Before starting, gather diagnostic context:

1. **Detect primary language** from config files (package.json, pyproject.toml, Cargo.toml, go.mod, Package.swift, Gemfile).
2. **Identify tooling** by searching for linter configs, formatter configs, build tool settings.
3. **Check project conventions** for existing patterns, module structure, and coding standards.
4. **Get scope overview** of the target area (if the user's provided input specifies a topic, scope to that; otherwise detect language from the project root).

Load the relevant references for the topic, based on the diagnostic context:

Reference library at `.agents/references/languages/`:
- async-deep-dive, async-patterns, bash-defensive-patterns
- browser-extension-development, cli-tool-development, cuda-gpu-programming
- fastapi-templates, go-concurrency-patterns, js-ts-patterns
- memory-management, nodejs-backend-patterns, pydantic-and-data-validation
- python-packaging-and-distribution, python-patterns, python-performance
- rust-project-patterns, swift-patterns, swift-performance
- testing-and-errors, uv-workflows

For deep language guidance, you may instead delegate to the `language-specialist` agent, passing the detected language/tooling and the request. It loads its skills (language-testing-patterns, test-driven-development) and the `.agents/references/languages/` library, then returns idiomatic guidance. Verify its output before presenting.

Help with: the user's provided input
