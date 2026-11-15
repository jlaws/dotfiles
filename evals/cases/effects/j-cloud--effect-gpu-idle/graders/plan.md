---
type: llm
focus: { source: file, path: gpu-plan.md }
---

All claims are required.
PASS if gpu-plan.md replaces always-on instances with capacity acquired per job (a job queue or scheduler that starts and releases GPUs).
PASS if spot use is paired with checkpointing and an interruption handler, so a reclaimed instance resumes instead of losing the run.
PASS if an idle-GPU alert is defined with a utilization threshold of 10 percent or a similarly low value and a stated time window.
FAIL if the plan recommends spot capacity without any checkpointing or resume mechanism.
