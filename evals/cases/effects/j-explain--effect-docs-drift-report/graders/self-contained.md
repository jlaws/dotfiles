---
type: regex
pattern: '<script\b[^>]*\bsrc\s*=|<link\b(?![^>]*\bhref\s*=\s*["'']?\s*data:)|<(?:img|iframe|frame|video|audio|source|embed|track|input)\b[^>]*\bsrc\s*=\s*["'']?\s*(?!data:|about:|blob:|#)[^\s"''>]|\bsrcset\s*=\s*["'']?\s*(?!data:)[^\s"''>]|<object\b[^>]*\bdata\s*=\s*["'']?\s*(?!data:)[^\s"''>]|<(?:image|use|feImage)\b[^>]*\bhref\s*=\s*["'']?\s*(?!data:|#)[^\s"''>]|<\w[^>]*\s(?:poster|background)\s*=\s*["'']?\s*(?!data:)[^\s"''>]|<meta\b[^>]*\bhttp-equiv\s*=\s*["'']?refresh|\bfetch\(|new\s+(?:XMLHttpRequest|WebSocket|EventSource)\b|new\s+(?:Shared)?Worker\s*\(\s*["''`]|\bsendBeacon\(|\.src\s*=\s*["''`]\s*(?!data:|blob:|#)[^\s"''`]|(?<![@\w.$])import\s*\(\s*["''`]|(?<![@\w.$])import\s*["''`]|(?<![@\w.$])import\b[^;\n]*?\bfrom\s*["''`]|@import\s*(?:url\(|["''])(?!\s*["'']?\s*data:)|image-set\(\s*["'']\s*(?!data:)|(?<!new\s)\burl\(\s*(?!["'']?\s*(?:data:|#|blob:|\)))["'']?\s*[^\s"'')]'
flags: i
match: not_contains
target: { source: file, path: drift-report.html }
---
