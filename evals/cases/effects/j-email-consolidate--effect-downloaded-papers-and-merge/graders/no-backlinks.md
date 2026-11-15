---
type: regex
pattern: '\]\(\s*<?[^)\s]*(?:emails/|\d{4}-\d{2}-\d{2}-[\w-]+\.md)'
match: not_contains
target: { source: file, path: digest.md }
---
