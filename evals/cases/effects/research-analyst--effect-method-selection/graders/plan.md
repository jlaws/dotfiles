---
type: llm
focus: { source: file, path: stats-plan.md }
---

All claims are required.
PASS if stats-plan.md picks a paired non-parametric test for analysis 1: a paired permutation test or a Wilcoxon signed-rank test.
PASS if stats-plan.md picks the Friedman test for analysis 2, followed by the Nemenyi post-hoc test for the pairwise differences.
PASS if stats-plan.md corrects analysis 3 with Benjamini-Hochberg (false discovery rate control).
PASS if stats-plan.md uses a Wilson interval for analysis 4.
FAIL if stats-plan.md picks a t-test (paired or unpaired) as the test for analysis 1, or a Bonferroni or Holm correction for analysis 3.
