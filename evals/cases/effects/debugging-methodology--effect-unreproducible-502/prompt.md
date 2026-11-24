---
description: "separate observed facts from inference, enumerate hypotheses with confirm and refute evidence, say when the cause is not established, and build a red-capable loop before any fix; .claude/skills/debugging-methodology/SKILL.md; hard because the log points so strongly at pool exhaustion that raising the pool size looks like the answer"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

We run a service called ledger-sync that is not in this checkout; only its logs are. Roughly one request in forty to POST /v1/sync returns a 502 at the gateway. Log excerpt:

```
10:02:11.204 gateway  req=a91 POST /v1/sync upstream=ledger-sync-2 status=200 dur=212ms
10:02:13.870 gateway  req=a94 POST /v1/sync upstream=ledger-sync-1 status=502 dur=30001ms
10:02:13.871 ledger-sync-1 worker=7 req=a94 acquiring db connection (pool 20/20 in use)
10:02:43.870 ledger-sync-1 worker=7 req=a94 db connection timeout after 30000ms
10:02:44.020 gateway  req=a97 POST /v1/sync upstream=ledger-sync-2 status=200 dur=198ms
10:02:51.442 ledger-sync-1 worker=3 req=a80 released db connection (held 41230ms)
10:03:02.500 gateway  req=a9c POST /v1/sync upstream=ledger-sync-1 status=502 dur=30002ms
```

Tell me the root cause and the fix. Save your diagnosis to ./diagnosis.md. In it, separate what the log shows from what you infer, list the candidate causes with the evidence that would confirm or refute each, say whether the root cause is established, and give the next step to take before applying any fix.
