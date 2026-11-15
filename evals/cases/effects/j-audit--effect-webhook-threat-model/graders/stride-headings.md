---
type: regex
pattern: '^#+\s*.*(Spoofing|Tampering|Repudiation|Information Disclosure|Denial of Service|Elevation of Privilege)'
flags: m
target: { source: file, path: threat-model.md }
---
