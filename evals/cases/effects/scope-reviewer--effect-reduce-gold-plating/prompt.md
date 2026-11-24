---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "recommend REDUCE when the scope gold-plates a narrow goal, keep the piece that serves the goal, and cut the speculative items in user-facing terms; .claude/agents/scope-reviewer.md; hard because each extra item sounds useful next to the badge"
---

Use the kb:scope-reviewer agent to review this scope before we design anything, and save the review to ./scope-review.md (the agent cannot write files, so the session saves its output). Start the file with the recommended mode (EXPAND, SELECTIVE EXPAND, HOLD, or REDUCE) and a one-line rationale, then list what to cut or keep in terms of user-facing outcomes.

Goal: let our 12 support agents see whether a customer has an open refund request before replying, to cut duplicate replies. Proposed scope for v1:
1. A badge in the existing ticket view showing "Refund open" with the request date.
2. A new refund-analytics dashboard with trend charts and forecasting.
3. A public REST API so partners can query refund status.
4. Role-based visibility rules with 6 configurable permission tiers.
5. Natural-language search over all refund history.
6. Email digests of open refunds to every agent each morning.
Timeline: 3 weeks, 2 engineers.
