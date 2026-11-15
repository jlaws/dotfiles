---
type: llm
focus: { source: file, path: claude-settings.json }
---

All claims are required.
PASS if permissions.allow lists the project's routine commands (such as running the tests with node --test) as exact invocations.
FAIL if any allow entry is a wildcard over a general-purpose runner or package manager (for example Bash(npm *), Bash(npx *), Bash(node *), or Bash(*)).
