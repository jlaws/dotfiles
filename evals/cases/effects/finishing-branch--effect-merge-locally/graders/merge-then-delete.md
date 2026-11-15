---
type: tool_order
before: { tool: Bash, input_match: 'git merge\s' }
after: { tool: Bash, input_match: 'git branch\s+(?:-[dD]\b|--delete\b)' }
---
