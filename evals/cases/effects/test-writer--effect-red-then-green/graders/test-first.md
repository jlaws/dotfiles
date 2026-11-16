---
type: tool_order
before: { tool: Write, input_match: '"file_path"\s*:\s*"[^"]*tests/test_empty\.py"' }
after: { tool: Edit, input_match: '"file_path"\s*:\s*"[^"]*calc/ops\.py"' }
---
