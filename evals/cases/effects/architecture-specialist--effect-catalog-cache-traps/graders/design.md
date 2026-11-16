---
type: llm
focus: { source: file, path: cache-design.md }
---

All claims are required.
PASS if cache-design.md spreads expiry with jittered TTLs (a random offset added to the TTL), not a single fixed TTL for every key.
PASS if cache-design.md prevents a stampede on a hot key with a mechanism such as a per-key lock or probabilistic early expiry, and warms critical keys at deploy time.
PASS if cache-design.md caches not-found results with a short TTL and uses a version prefix or namespace in key names so a value format change does not read old entries.
PASS if cache-design.md invalidates the cached page when the prices table changes (not only when products change), using event-driven or tag-based invalidation.
FAIL if cache-design.md gives every key the same fixed TTL or leaves price changes to expire naturally.
