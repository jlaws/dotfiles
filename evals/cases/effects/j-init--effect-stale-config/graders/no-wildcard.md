---
type: regex
target: { source: file, path: claude-settings.json }
pattern: 'Bash\([^)"]*\*'
match: not_contains
---
