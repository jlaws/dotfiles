---
type: regex
pattern: '^#+[^\n]*\b(?:rejected|dead[- ]ends?|alternatives?|considered)\b'
flags: im
target: { source: file, path: HANDOFF.md }
---
