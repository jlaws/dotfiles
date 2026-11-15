---
type: regex
pattern: '^#+\s*.*(Metrics|What Shipped|High-Churn)'
flags: mi
target: { source: file, path: retro.md }
---
