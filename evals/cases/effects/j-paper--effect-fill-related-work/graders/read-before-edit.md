---
type: tool_order
before: { tool: Read, input_match: 'main\.tex' }
after: { tool: Edit, input_match: '"file_path"\s*:\s*"[^"]*main\.tex"' }
---
