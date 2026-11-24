---
type: regex
target: { source: file, path: plans/plan.md }
pattern: '^(?=[^\n]*\|)[^\n]*\bRisks?\b[^\n]*\n[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\||$)'
flags: mi
---
