---
type: regex
pattern: 'def find_user\([^)]*\):(?:(?!\ndef )[\s\S])*?execute\(\s*[\"''''][^\"'''']*(?:\?|:\w+)[^\"'''']*[\"'''']\s*,'
target: { source: file, path: app/users.py }
---
