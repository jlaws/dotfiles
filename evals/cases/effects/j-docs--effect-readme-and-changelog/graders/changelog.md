---
type: llm
focus: { source: file, path: CHANGELOG.md }
---

All claims are required.
PASS if CHANGELOG.md has an Unreleased section above a release heading for 1.0.0 dated 2026-11-14 in ISO format.
PASS if any entry about the README or flag name is filed under Fixed (or omitted), because documentation corrected a flag that never worked, so it is not Removed and not a breaking change.
FAIL if CHANGELOG.md has a Breaking, Removed, or Migration section describing the --out correction.
