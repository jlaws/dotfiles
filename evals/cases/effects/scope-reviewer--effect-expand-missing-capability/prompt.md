---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:scope-reviewer agent to review this scope before design starts, and save the review to ./scope-review.md (the agent cannot write files, so the session saves its output). Start the file with the recommended mode (EXPAND, SELECTIVE EXPAND, HOLD, or REDUCE) and a one-line rationale. If you recommend adding anything, name the specific capability and the user-facing outcome it protects; if you recommend leaving the scope alone, say why.

Goal: customers can cancel their subscription themselves and never be billed again afterwards, so support tickets about "I cancelled but was charged" drop to near zero.
Proposed scope for v1:
1. A "Cancel subscription" button in account settings with a confirmation dialog.
2. Cancelling sets the subscription status to cancelled in our database.
3. A confirmation email saying the subscription is cancelled.
Not in scope: anything involving the payment provider, because "the provider handles billing on its own schedule".
Timeline: 2 weeks, 2 engineers.
