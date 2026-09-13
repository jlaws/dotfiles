---
name: cmd-j-init
description: "Use when invoking the j-init workflow."
disable-model-invocation: true
---

# Init

Invoke the `project-scaffolding` skill before doing anything else. It owns the method: what to read
before asking, what to interview for, where version numbers come from, and what gets written.

For stack idioms and project layout you may delegate to `language-specialist` (loads
`references/languages/`). Verify its output, and never take a version number from it.

Target: the user's provided input. If none was provided, scaffold the current working directory.
