---
type: llm
focus: { source: file, path: training-plan.md }
---

All claims are required.
PASS if training-plan.md states a split rule that prevents leakage across the split: a time-based split (older tickets train, newer test), a customer-grouped split (no customer in both train and test), or a time split that also holds out customers; random shuffling of tickets before splitting is not accepted.
PASS if training-plan.md fits TF-IDF on the training split only and confines any rebalancing (oversampling or class weights) to training, leaving validation and test at the natural 5 percent rate.
PASS if training-plan.md replaces accuracy with a metric suited to imbalance (macro F1, or precision and recall on the urgent class, or PR-AUC).
PASS if training-plan.md quantifies uncertainty on the reported number with a bootstrap confidence interval or equivalent.
FAIL if training-plan.md keeps accuracy as the headline metric or oversamples before splitting.
