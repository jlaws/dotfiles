---
type: regex
pattern: '\.unwrap\s*\(|\.expect\s*\('
match: not_contains
target: { source: file, path: main.rs }
---
