---
type: llm
focus: { source: file, path: pagination.md }
---

All claims are required.
PASS if pagination.md chooses cursor-based or keyset pagination (not offset/limit) for the feed.
PASS if the cursor or sort key includes a unique tiebreaker (such as an id) in addition to created_at, so items sharing a timestamp are neither skipped nor repeated.
PASS if the rejected alternative is offset/limit (or page numbers) and the reason names skipped or duplicated items when rows are inserted during scrolling.
FAIL if pagination.md states no maximum page size, or sorts on created_at alone.
