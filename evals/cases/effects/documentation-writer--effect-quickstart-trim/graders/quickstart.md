---
type: llm
focus: { source: file, path: docs/quickstart.md }
---

All claims are required.
PASS if docs/quickstart.md has 3 to 5 steps (a prerequisites list and a next-steps section do not count as steps).
PASS if docs/quickstart.md states, before the first step, the concrete result the reader will have at the end of the guide.
FAIL if a code block in docs/quickstart.md keeps an elision standing in for code, such as the draft's "# ... handle errors here" line or "events=[...]", or uses os.environ without importing os.
