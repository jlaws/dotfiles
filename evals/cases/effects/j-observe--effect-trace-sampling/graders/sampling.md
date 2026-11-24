---
type: llm
focus: { source: file, path: sampling.md }
---

All claims are required.
PASS if sampling.md sets production probabilistic sampling at a rate from 10% to 50%.
PASS if downstream services follow the sampling decision of the incoming parent span (a ParentBased sampler or equivalent), so a trace is kept or dropped as a whole.
PASS if traces of failed requests and of high-latency requests are kept regardless of the probabilistic rate (for example, by tail sampling in the Collector).
FAIL if the production probabilistic rate is below 10% or above 50%.
