---
type: llm
focus: { source: file, path: analysis.md }
---

All claims are required.
PASS if the analysis flags the exactly 2.00x speedup and the +- 0.00 spread (a single run per configuration) as implausible or unsupported statistics.
PASS if the analysis says the ablation removes only the 16-step warmup, not the routing mechanism, so it does not support the claim that the router drives the gains.
PASS if the analysis notes the routing threshold was tuned on the evaluation corpus while baselines were left untuned.
FAIL if the overall reviewer score is 7 or higher.
