---
name: j-docs
description: "Documentation consultation — technical writing, API docs, and changelogs. Use when writing docs, generating OpenAPI specs, or automating changelogs. Do NOT use for inline code comments (write directly)."
argument-hint: "<question-or-task>"
model: sonnet
effort: medium
---

Load skill `analysis-output-patterns` for output structure rules.

Before starting, gather diagnostic context:

1. **Detect documentation tooling** from config files (docusaurus.config.js, mkdocs.yml, .readthedocs.yml, typedoc.json, sphinx conf.py).
2. **Identify existing docs** by searching for docs/, README.md, CHANGELOG.md, or API specification files.
3. **Check for API schemas** by searching for openapi.yaml, swagger.json, or GraphQL schema files.
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a component, scope to that; otherwise scan for documentation directories).

For deep documentation guidance, delegate to the `documentation-writer` agent via the Task tool, passing the diagnostic findings above and the request. It loads its skills (post-ship-doc-sync, output-completeness) and the `references/documentation/` library, then returns specific guidance. Verify its output before presenting.

Help with: $ARGUMENTS
