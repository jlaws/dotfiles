---
type: regex
pattern: '(?:\b(?:db|export|config)\.py\b|\bfind_orders\b|\barchive\b|\bPAYMENT_API_SECRET\b)[\s\S]{0,400}?\bUNVERIFIED\b|\bUNVERIFIED\b[\s\S]{0,400}?(?:\b(?:db|export|config)\.py\b|\bfind_orders\b|\barchive\b|\bPAYMENT_API_SECRET\b)'
target: { source: file, path: security-report.md }
---
