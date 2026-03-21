---
name: cmd-documentation-docs
description: "Documentation consultation — technical writing, API docs, and changelogs. Use when writing docs, generating OpenAPI specs, or automating changelogs. Do NOT use for inline code comments (write directly)."
disable-model-invocation: true
---

# Documentation Consultation

Before starting, gather diagnostic context:

1. **Detect documentation tooling** from config files (docusaurus.config.js, mkdocs.yml, .readthedocs.yml, typedoc.json, sphinx conf.py).
2. **Identify existing docs** by searching for docs/, README.md, CHANGELOG.md, or API specification files.
3. **Check for API schemas** by searching for openapi.yaml, swagger.json, or GraphQL schema files.
4. **Get scope overview** of the target area (if the user specifies a component, scope to that; otherwise scan for documentation directories).

Help with the documentation topic specified by the user.
