---
type: regex
pattern: 'export\s+let\b|^\s*\$:|\bon:click\b|<slot\b|createEventDispatcher'
flags: m
match: not_contains
target: { source: file, path: Cart.svelte }
---
