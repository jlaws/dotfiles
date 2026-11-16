---
type: regex
pattern: 'def find_user\([^)]*\)[^\n]*:(?:(?!\ndef )[\s\S])*?(?:execute\(\s*["''][^"''\n]*SELECT[^"''\n]*(?:\?|:\w+)[^"''\n]*["'']\s*,|["''][^"''\n]*SELECT[^"''\n]*(?:\?|:\w+)[^"''\n]*["''](?:(?!\ndef )[\s\S])*?execute\(\s*\w+\s*,)'
flags: i
target: { source: file, path: app/users.py }
---
