---
type: llm
focus: { source: file, path: pagination.md }
---

All claims are required.
PASS if pagination.md chooses cursor-based or keyset pagination (not offset/limit) for the feed.
PASS if the cursor or sort key includes a unique tiebreaker (such as an id) in addition to created_at, so items sharing a timestamp are neither skipped nor repeated.
PASS if pagination.md names at least one rejected alternative with a reason that holds for a feed where constant inserts shift item positions (for example offset/limit or page numbers skipping or repeating items, or a created_at-only cursor skipping or repeating items that share a timestamp).
FAIL if pagination.md chooses offset/limit or page numbers, states no maximum page size, or sorts on created_at alone.
