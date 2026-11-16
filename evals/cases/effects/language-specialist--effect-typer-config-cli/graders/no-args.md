---
type: regex
pattern: '^app(?:\s*:\s*[\w.]+)?\s*=\s*(?:typer\.)?Typer\([^)]*no_args_is_help\s*=\s*True|^@app\.callback\([^)]*invoke_without_command\s*=\s*True[\s\S]*\.get_help\(\)'
flags: m
target: { source: file, path: mytool.py }
---
