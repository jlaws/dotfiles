---
type: tool_order
before: { tool: Bash, input_match: 'git\s+reset\s+--soft' }
after: { tool: Bash, input_match: 'git\s+rebase\s' }
---
