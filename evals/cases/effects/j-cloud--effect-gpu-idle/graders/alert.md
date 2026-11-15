---
type: regex
pattern: '^(?=[^\n]*(?:utili[sz]ation|idle))[^\n]*(?:<=?|\bbelow|\bunder|\bless\s+than|\blower\s+than)\s*\d+(?:\.\d+)?\s*(?:%|percent)'
flags: im
target: { source: file, path: gpu-plan.md }
---
