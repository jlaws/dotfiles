---
type: regex
pattern: '\bserial\b|\bbigserial\b|\bmoney\b|\bvarchar\s*\(|\bchar\s*\(|\btimestamp\s*(?:\(|,|\n|$|not|default)'
flags: im
match: not_contains
target: { source: file, path: schema.sql }
---
