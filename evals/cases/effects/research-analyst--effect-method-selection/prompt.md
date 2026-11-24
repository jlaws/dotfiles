---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "pick the method from the decision tables: a paired non-parametric test for paired seeds, the Friedman test with the Nemenyi post-hoc for many models across datasets, Benjamini-Hochberg for an exploratory ablation, and a Wilson interval for an accuracy proportion; .claude/references/research/statistical-analysis.md; hard because a paired t-test, a Holm or Bonferroni correction, and a normal-approximation or bootstrap interval are each common defaults"
---

Use the kb:research-analyst agent to choose the statistical method for each of four analyses in our paper. Use only the material in this prompt; do not search the web.
1. Our model against one baseline, both trained with the same 10 seeds on the same data splits. We have one macro-F1 per seed for each system and want to know whether the difference is real.
2. Five models evaluated on 14 benchmark datasets, one score per model per dataset. We want to claim an overall ranking and say which pairs of models differ.
3. An exploratory ablation: 4 components switched on and off in all 16 combinations, each reduced combination compared with the full model, to report which removals matter.
4. A confidence interval for one model's accuracy on a test set of 2,000 examples.

For each analysis, name the test or interval method, say whether a multiple-comparison correction is needed and which one, and give the reason. Save the choices to ./stats-plan.md.
