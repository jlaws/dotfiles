---
type: llm
focus: { source: file, path: pyproject.toml }
---

All claims are required.
PASS if the httpx requirement in pyproject.toml keeps a lower bound and has no upper bound (for example httpx>=0.25).
PASS if the markdown-it-py requirement still excludes 4.0 and later (for example markdown-it-py>=3.0,<4), since docs/compat.md records that 4.0 breaks the library.
FAIL if httpx keeps any upper cap (such as <1.0, <1, or ~=0.25), or the markdown-it-py cap below 4 is removed.
