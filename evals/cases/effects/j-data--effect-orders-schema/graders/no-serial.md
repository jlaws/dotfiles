---
type: regex
pattern: '(?:^|[(,])\s*"?\w+"?\s+(?:(?:small|big)?serial\b|money\b|(?:var)?char(?:acter)?(?:\s+varying)?\s*\(|timestamp\b(?!\s*(?:\(\d\)\s*)?with\s+time\s+zone))'
flags: im
match: not_contains
target: { source: file, path: schema.sql }
---
