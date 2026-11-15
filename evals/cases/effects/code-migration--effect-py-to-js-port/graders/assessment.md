---
type: llm
focus: { source: file, path: migration-assessment.md }
---

All claims are required.
PASS if the scope section gives measured numbers from this repo (file counts or lines), not estimates.
PASS if the unresolved questions flag that mean([]) raises ZeroDivisionError in the code while the docstring promises 0.0, and ask which behavior the port should preserve instead of silently choosing one.
PASS if exactly one strategy is chosen (not a combination) with a reason tied to the size of this codebase.
FAIL if the complexity is a single averaged score or number with no factor named as the driver.
