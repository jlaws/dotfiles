---
type: llm
focus: { source: file, path: draft/agents/dep-auditor.md }
---

All claims are required.
PASS if the description states what the agent is and when to use it and ends with a Do NOT use for clause.
PASS if the body opens with a role statement starting You are a and does not inline a long methodology.
FAIL if the frontmatter sets a memory key or the tools include Write, Edit, or Bash.
