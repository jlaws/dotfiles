---
type: regex
pattern: 'await\s+(?:props\.)?params|params\s*=\s*await|Promise<\{\s*id'
flags: i
target: { source: file, path: page.tsx }
---
