---
type: llm
focus: { source: file, path: pyproject.toml }
---

All claims are required.
PASS if build-system requires include both hatchling and hatch-vcs (or another git-tag version plugin for hatchling), and the version is declared dynamic with the vcs version source configured.
PASS if the httpx requirement has no upper bound (no <, no ~=, no == pin) and pydantic is required at 2 or newer.
FAIL if a literal version string is set under [project].
