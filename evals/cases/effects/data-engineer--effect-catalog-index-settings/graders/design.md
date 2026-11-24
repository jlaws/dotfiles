---
type: llm
focus: { source: file, path: search-index.md }
---

All claims are required.
PASS if search-index.md sets between 12 and 30 primary shards for the 600 GB, sized so each primary shard holds 20 to 50 GB.
PASS if the mapping in search-index.md sets "dynamic": "strict" at the top level (the supplier attributes may still go in a flattened or explicitly mapped field).
FAIL if the bulk-load settings in search-index.md leave refresh_interval at its 1-second default or any other positive interval, instead of disabling it ("-1") and refreshing once the load ends.
