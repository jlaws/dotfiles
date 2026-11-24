---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "split by time or customer before any fitting, fit TF-IDF and rebalance on train only, replace accuracy with an imbalance-aware metric, and give a bootstrap interval; .claude/agents/ml-engineer.md; hard because the plan's order of steps leaks in several places at once, and high accuracy on a 5 percent class looks like a strong result"
---

Use the kb:ml-engineer agent to fix the training plan below and save the corrected plan to ./training-plan.md. Current plan: "We have 200,000 support tickets from 8,000 customers over 18 months, labeled urgent or not; 5 percent are urgent. We shuffle all tickets, fit TF-IDF on the whole set, oversample urgent tickets to 50 percent, then split 80/10/10 into train, validation, and test, train a classifier, and report accuracy on the test set." The corrected plan must list every flaw with its fix, give the new order of steps, state the split rule, name the headline metric, and say how the uncertainty on the reported number is quantified.
