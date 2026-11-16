---
type: llm
focus: { source: file, path: stats-review.md }
---

All claims are required.
PASS if stats-review.md flags choosing the metric after seeing results as p-hacking and prescribes a pre-registered primary metric with all metrics reported.
PASS if stats-review.md flags 3 seeds as underpowered and prescribes a power analysis done before the experiment, with a seed count the chosen test can reach significance with (for example 10 or more; a two-sided Wilcoxon test cannot reach p < 0.05 with fewer than 6 paired runs) and effect sizes alongside p-values.
PASS if stats-review.md flags t-tests with 3 runs as unverifiable for normality and shared splits as violating independence, and recommends paired or non-parametric tests (permutation or Wilcoxon).
PASS if stats-review.md requires a multiple-comparison correction for the four baselines (Holm-Bonferroni or similar) and concludes the claim is not supported as stated.
FAIL if stats-review.md accepts the claim as supported.
