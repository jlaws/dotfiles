---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

You were dispatched by an orchestrator that will paste your reply verbatim into its own context and cannot ask you follow-ups. Your task: review this function and return your findings.

def mean(xs):
    return sum(xs) / len(xs)
