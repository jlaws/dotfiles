---
description: "Security threat model and vulnerability scan — STRIDE analysis, SAST patterns, and compliance mapping. Use when reviewing code for vulnerabilities, conducting threat modeling, or mapping compliance controls."
---

Before invoking the skill, perform reconnaissance:

1. **Detect project language/framework** from config files (package.json, Cargo.toml, go.mod, requirements.txt, Gemfile, etc.).
2. **Identify auth patterns** by searching for auth-related imports and middleware.
3. **Check existing security tooling** config (.semgrep.yml, .snyk, .trivyignore, .eslintrc security plugins).
4. **Get file tree overview** of the target scope (if $ARGUMENTS specifies a component/directory, scope to that; otherwise scope to the full project).

Read references/security/security-analysis.md and apply its methodology to analyze: $ARGUMENTS
