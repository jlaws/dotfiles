---
type: llm
focus: { source: file, path: CHANGELOG.md }
---

All claims are required.
PASS if CHANGELOG.md has Breaking entries for the Config to WidgetConfig rename and the /v1 to /v2 base URL change, and a Migration section with concrete before and after steps (for example Config to WidgetConfig in imports, /v1/ to /v2/ in base URLs).
PASS if CHANGELOG.md lists stream() under Added, the race condition and timeout default under Fixed with issue numbers #298 and #301, and poll() as Deprecated with removal in 3.0.
FAIL if CHANGELOG.md mentions the eslint bump, the CI cache change, the merge commit, the http.js split, or the contributing-guide typo.
