---
type: tool_order
before: { tool: Edit, input_match: '"file_path"\s*:\s*"[^"]*test/[^"]*\.test\.js"' }
after: { tool: Edit, input_match: '"file_path"\s*:\s*"[^"]*src/todo\.js"' }
---
