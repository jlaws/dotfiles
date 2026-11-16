---
type: regex
pattern: '\[blocking\](?:(?!\[(?:blocking|important|nit|suggestion)\])[\s\S]){0,250}?(?:auth\.py|signature|current_user|\btoken\b|HMAC)|(?:auth\.py|current_user|signature)[^\n]*\[blocking\]'
flags: i
target: { source: file, path: review.md }
---
