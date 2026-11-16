---
type: llm
focus: { source: file, path: notes.md }
---

All claims are required.
PASS if notes.md says the CI checkout needs full git history and tags (for example actions/checkout with fetch-depth: 0), because a shallow clone produces a wrong fallback version.
PASS if notes.md names py.typed (src/acme_geo/py.typed) as the extra file.
FAIL if notes.md suggests setting a fixed version string in pyproject.toml as the fix for the CI problem.
