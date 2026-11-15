---
type: regex
pattern: '^(?=[\s\S]*generated\s+(?:always|by\s+default)\s+as\s+identity)(?=[\s\S]*(?:timestamptz|timestamp(?:\s*\(\d\))?\s+with\s+time\s+zone))(?=[\s\S]*numeric)'
flags: i
target: { source: file, path: schema.sql }
---
