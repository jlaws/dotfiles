---
type: llm
focus: { source: file, path: training-plan.md }
---

All claims are required.
PASS if training-plan.md splits by customer (grouped) so no customer appears in both train and test, and splits by time (older tickets train, newer test) because the data spans 18 months.
PASS if training-plan.md fits TF-IDF on the training split only and oversamples urgent tickets only inside the training split, leaving validation and test at the natural 5 percent rate.
PASS if training-plan.md replaces accuracy with a metric suited to imbalance (macro F1, or precision and recall on the urgent class, or PR-AUC).
PASS if training-plan.md quantifies uncertainty on the reported number with a bootstrap confidence interval or equivalent.
FAIL if training-plan.md keeps accuracy as the headline metric or oversamples before splitting.
