---
description: "Security threat model and vulnerability scan — STRIDE analysis, SAST patterns, and compliance mapping."
---

Before invoking the skill, perform reconnaissance:

1. **Detect project language/framework** from config files (package.json, Cargo.toml, go.mod, requirements.txt, Gemfile, etc.).
2. **Identify auth patterns** by searching for auth-related imports and middleware.
3. **Check existing security tooling** config (.semgrep.yml, .snyk, .trivyignore, .eslintrc security plugins).
4. **Get file tree overview** of the target scope (if $ARGUMENTS specifies a component/directory, scope to that; otherwise scope to the full project).

Invoke the security:security-analysis skill and use it to analyze: $ARGUMENTS
