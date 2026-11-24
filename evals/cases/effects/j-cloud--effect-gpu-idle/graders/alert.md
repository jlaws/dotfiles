---
type: regex
pattern: 'GPU utili[sz]ation\s*(?:<=?|≤)\s*10(?:\.0+)?\s*%\s*(?:for|over)\s+\d+(?:\.\d+)?\s*(?:h|hrs?|hours?|m|mins?|minutes?)\b'
flags: i
target: { source: file, path: gpu-plan.md }
---
