---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "generate a concrete option under each of the four modes and identify the market, technical, resource, and timeline assumptions before recommending a mode, and do not expand scope without new evidence; .claude/references/product/scope-review-methodology.md; hard because a reviewer reaches a verdict and argues for it, so the modes it rejects get no options and the assumptions stay implicit"
---

Use the kb:scope-reviewer agent to review this scope before design starts, and save the review to ./scope-review.md (the agent cannot write files, so the session saves its output). Start the file with the recommended mode (EXPAND, SELECTIVE EXPAND, HOLD, or REDUCE) and a one-line rationale, then show the work behind it: the options you weighed and the assumptions the scope rests on.

Goal: clients of our 40-person design agency approve deliverables online instead of by email, so a proof is approved within 2 days instead of the current 6.
Proposed scope for v1:
1. A client portal with email-and-password login.
2. Upload of proofs (PDF and PNG, up to 200 MB each) by the agency's designers.
3. Approve and Request changes buttons on each proof.
4. Comment threads on each proof.
5. Email notifications when a proof is uploaded or a decision is made.
The CEO also wants invoicing and time tracking in the portal because competitors have them; no client has asked for either.
Timeline: 6 weeks, 2 engineers, one of whom joins in week 3.
