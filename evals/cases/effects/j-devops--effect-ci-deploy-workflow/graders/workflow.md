---
type: llm
focus: { source: file, path: ci.yml }
---

All claims are required.
PASS if every uses: line for a third-party action is pinned to a 40-hex-character SHA (a placeholder SHA with a comment counts) rather than a tag like v4.
PASS if every job declares its own permissions block with read-only or minimal scopes, and the deploy job is limited to pushes on main and needs the test job.
PASS if a concurrency group covers the deploy and cancel-in-progress is false or absent for it.
FAIL if any job relies on the default write-all token, or cancel-in-progress: true applies to the deploy job.
