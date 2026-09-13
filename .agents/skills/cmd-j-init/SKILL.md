---
name: cmd-j-init
description: "Use when invoking the j-init workflow."
disable-model-invocation: true
---

# Init

Invoke the `project-scaffolding` skill before doing anything else.

Target: the user's provided input. If none was provided, scaffold the current working directory.

Read the repository before asking anything. If it has a README or code, infer purpose and stack and
ask the user to correct you. If it is empty, invoke the `design-first` skill to establish purpose,
requirements, and constraints first.

For stack idioms and project layout you may delegate to `language-specialist` (loads
`references/languages/`). Verify its output, and never take a version number from it: version
currency comes from a registry query.
