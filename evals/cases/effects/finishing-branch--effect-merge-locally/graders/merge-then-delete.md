---
type: tool_order
before: { tool: Bash, input_match: 'git merge' }
after: { tool: Bash, input_match: 'git branch -d' }
---
