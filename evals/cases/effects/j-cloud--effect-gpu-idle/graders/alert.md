---
type: regex
pattern: '(?:utili[sz]ation|idle)[^\n]*\d+\s*%|\d+\s*%[^\n]*(?:utili[sz]ation|idle)'
flags: i
target: { source: file, path: gpu-plan.md }
---
