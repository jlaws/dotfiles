---
type: llm
focus: { source: file, path: draft/skills/log-triage/SKILL.md }
---

All claims are required.
PASS if the frontmatter description gives triggering conditions and does not summarize the workflow steps, and no angle bracket character appears in the frontmatter.
PASS if the frontmatter has no model key and the body has headings for When to use and Gotchas.
FAIL if the body contains a rationalization table, a Red Flags STOP list, or an all-caps non-negotiable rule.
