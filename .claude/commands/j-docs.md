---
name: j-docs
description: "Documentation consultation — technical writing, API docs, and changelogs. Use when writing docs, generating OpenAPI specs, or automating changelogs. Do NOT use for inline code comments (write directly)."
argument-hint: "<question-or-task>"
---

Load skill `analysis-output-patterns` for output structure rules.
Load skill `output-completeness` to avoid truncated or stubbed documentation.

Before starting, gather diagnostic context:

1. **Detect documentation tooling** from config files (docusaurus.config.js, mkdocs.yml, .readthedocs.yml, typedoc.json, sphinx conf.py).
2. **Identify existing docs** by searching for docs/, README.md, CHANGELOG.md, or API specification files.
3. **Check for API schemas** by searching for openapi.yaml, swagger.json, or GraphQL schema files.
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a component, scope to that; otherwise scan for documentation directories).

Load relevant references based on the diagnostic context:
- `references/documentation/technical-writing-for-devtools` -- writing style, structure, developer-audience docs
- `references/documentation/readme-template` -- README structure and sections
- `references/documentation/quickstart-template` -- getting-started/tutorial structure
- `references/documentation/api-doc-template` -- reference-doc structure for APIs
- `references/documentation/openapi-spec-generation` -- generating/maintaining OpenAPI specs
- `references/documentation/changelog-patterns`, `changelog-automation` -- changelog conventions and automated release notes

Help with: $ARGUMENTS
