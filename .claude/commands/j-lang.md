---
name: j-lang
description: "Language patterns consultation — idioms, tooling, and project scaffolding. Use when setting up a project, choosing patterns, or working in Python, JS/TS, Go, Rust, Swift, or other languages. Do NOT use for framework-specific help (use /j-frontend or domain specialist instead)."
argument-hint: "<question-or-task>"
---

Load skill `analysis-output-patterns` for output structure rules.
Load skill `language-testing-patterns` for language-idiomatic test guidance.

Before starting, gather diagnostic context:

1. **Detect primary language** from config files (package.json, pyproject.toml, Cargo.toml, go.mod, Package.swift, Gemfile).
2. **Identify tooling** by searching for linter configs, formatter configs, build tool settings.
3. **Check project conventions** for existing patterns, module structure, and coding standards.
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a topic, scope to that; otherwise detect language from the project root).

Load relevant references for the detected language/topic:
- **Python**: `references/languages/python-patterns`, `python-performance`, `python-packaging-and-distribution`, `pydantic-and-data-validation`, `fastapi-templates`, `uv-workflows` -- idioms, perf, packaging, validation, web, uv tooling
- **JS/TS & Node**: `references/languages/js-ts-patterns`, `nodejs-backend-patterns`, `browser-extension-development` -- language idioms, backend patterns, extensions
- **Go / Rust / Swift**: `references/languages/go-concurrency-patterns`, `rust-project-patterns`, `swift-patterns`, `swift-performance` -- language-specific structure and concurrency
- **Cross-cutting**: `references/languages/async-patterns`, `async-deep-dive`, `concurrency-patterns`, `memory-management`, `testing-and-errors` -- async/concurrency, memory, testing & error handling
- **Systems & CLI**: `references/languages/bash-defensive-patterns`, `cli-tool-development`, `cuda-gpu-programming` -- robust shell, CLI design, GPU programming

Help with: $ARGUMENTS
