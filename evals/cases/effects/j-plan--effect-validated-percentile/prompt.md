---
tags: [effects, effect, heavy]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "follow the writing-plans phase skeleton, with branch and docs first, test-first phases of one commit each, a final validation gate, PR boundaries, a risks table, and no scope beyond the spec; .claude/commands/j-plan.md; hard because the redesign gate bars features the spec did not ask for, and nearest-rank percentile invites interpolation modes and other statistics"
---

/kb:j-plan plan a new function percentile(xs, p) for calc/ops.py: nearest-rank percentile of a list, raising ValueError for an empty list and for p outside 0 to 100. Plan mode is not available in this session, so use ./plans/plan.md as the plan file in its place; plans/ is git-ignored here, so the plan stays out of the repository. Requirements for the plan: it must not start work on main (the first phase opens a branch off origin/main) and that same first phase writes the documentation of the new behavior in README.md before any code; the code phases are test-first and each ends in one commit; the last phase is a validation gate that states the gate command or says explicitly that no benchmark applies and why; the plan states its pull request boundaries; it has a Risks table where each risk names the evidence that closes it; and it contains no open-questions section.
