---
type: regex
pattern: '<script\b[^>]*\bsrc\s*=|<link\b(?![^>]*\bhref\s*=\s*["'']?\s*data:)|<(?:img|iframe|video|audio|source|embed|track)\b[^>]*\bsrc\s*=\s*["'']?\s*(?:https?:)?//|\bsrcset\s*=\s*["'']?\s*(?!data:)[^\s"''>]|<object\b[^>]*\bdata\s*=\s*["'']?\s*(?!data:)[^\s"''>]|<(?:image|use|feImage)\b[^>]*\bhref\s*=\s*["'']?\s*(?!data:|#)[^\s"''>]|<meta\b[^>]*\bhttp-equiv\s*=\s*["'']?refresh|\bfetch\(|new\s+(?:XMLHttpRequest|WebSocket|EventSource)\b|\bsendBeacon\(|@import\s+(?:url\(\s*["'']?\s*(?!data:)|["''](?!data:))|url\(\s*["'']?\s*(?:https?:)?//'
flags: i
match: not_contains
target: { source: file, path: calc-explainer.html }
---
