---
type: regex
pattern: '^(?=[\s\S]*^[ \t]*(?:\*\*/)?\.git/?[ \t]*$)(?=[\s\S]*^[^#\n]*(?:venv|^[ \t]*(?:\*\*/)?\.?env/?[ \t]*$))(?=[\s\S]*^[^#\n]*(?:cache|\.py[cod]))(?=[\s\S]*^[ \t]*(?:\*\*/)?tests?(?:/\*{0,2})?[ \t]*$)'
flags: m
target: { source: file, path: .dockerignore }
---
