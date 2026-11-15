---
tags: [effects, effect, heavy]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-plan plan a new function percentile(xs, p) for calc/ops.py: nearest-rank percentile of a list, raising ValueError for an empty list and for p outside 0 to 100. Save the plan to ./plans/plan.md. Requirements for the plan: it must not start work on main (the first phase opens a branch off origin/main) and that same first phase writes the documentation of the new behavior in README.md before any code; the code phases are test-first and each ends in one commit; the last phase is a validation gate that states the gate command or says explicitly that no benchmark applies and why; the plan states its pull request boundaries; it has a Risks table where each risk names the evidence that closes it; and it contains no open-questions section.
