---
type: llm
focus: { source: file, path: design.md }
---

All claims are required.
PASS if design.md compares at least two distinct approaches (for example nearest-rank versus linear interpolation) with trade-offs and names one as the recommendation.
PASS if design.md has a numbered list of assumptions and each is falsifiable (a concrete claim about inputs, ranges, or behavior, not a vague wish).
PASS if design.md points out that calc/ops.py mean() documents 0.0 for an empty list while its body would raise ZeroDivisionError, and ties that conflict to the empty-list decision for percentile.
FAIL if design.md recommends behavior without stating assumptions, or presents only one approach.
