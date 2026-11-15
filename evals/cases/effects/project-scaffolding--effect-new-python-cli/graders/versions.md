---
type: llm
focus: { source: file, path: pyproject.toml }
---

All claims are required.
PASS if no dependency is pinned to an exact version that was asserted from memory as the current release; either the manifest uses ranges, or each pinned version is marked as unverified in a comment.
FAIL if the manifest pins exact versions with no indication that they were checked against a registry.
