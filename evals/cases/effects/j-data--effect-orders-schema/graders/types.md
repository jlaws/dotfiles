---
type: regex
pattern: '^(?=[\s\S]*generated\s+always\s+as\s+identity)(?=[\s\S]*timestamp(?:tz|\s+with\s+time\s+zone))(?=[\s\S]*numeric)'
flags: i
target: { source: file, path: schema.sql }
---
