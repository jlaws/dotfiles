---
type: llm
focus: { source: file, path: calc/stats.py }
---

All claims are required.
PASS if a marker comment that records the scan as a deliberate simplification sits on or directly above the top_k scan.
PASS if that comment names a ceiling, meaning what breaks and roughly when (for example a large list or a large k making the scan too slow).
PASS if that comment names an upgrade path (for example heapq.nlargest or a sort).
FAIL if any bare TODO or FIXME without a ticket and reason appears, or if top_k or median is missing.
