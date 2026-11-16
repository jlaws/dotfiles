---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:ml-engineer agent to fix the training plan below and save the corrected plan to ./training-plan.md. Current plan: "We have 200,000 support tickets from 8,000 customers over 18 months, labeled urgent or not; 5 percent are urgent. We shuffle all tickets, fit TF-IDF on the whole set, oversample urgent tickets to 50 percent, then split 80/10/10 into train, validation, and test, train a classifier, and report accuracy on the test set." The corrected plan must list every flaw with its fix, give the new order of steps, state the split rule, name the headline metric, and say how the uncertainty on the reported number is quantified.
