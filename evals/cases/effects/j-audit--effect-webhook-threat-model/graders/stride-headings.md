---
type: regex
pattern: '^(?=[\s\S]*^#+[^\n]*Spoofing)(?=[\s\S]*^#+[^\n]*Tampering)(?=[\s\S]*^#+[^\n]*Repudiation)(?=[\s\S]*^#+[^\n]*Information\s+Disclosure)(?=[\s\S]*^#+[^\n]*Denial\s+of\s+Service)(?=[\s\S]*^#+[^\n]*Elevation\s+of\s+Privilege)'
flags: im
target: { source: file, path: threat-model.md }
---
