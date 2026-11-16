---
type: regex
pattern: '^API_TOKEN=not-a-real-token$'
flags: m
target: { source: file, path: env.txt }
---
