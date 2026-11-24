---
description: "a finding is pre-existing only when no hunk introduces, exposes, or makes it stale, so a latent bug the change newly exposes is this change's finding; .claude/skills/code-review-patterns/SKILL.md; hard because to_cents sits in an unchanged file and a model tags anything outside the hunks as pre-existing"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Review the change in ./change.diff for its author and save the review to ./review.md. The working tree in this directory already has the change applied, so read the surrounding code as you need. Requirements: every finding cites file:line and carries a priority P0 to P3; tag a finding `(pre-existing)` when this change is not responsible for it, and leave the tag off when it is; no praise; end with a single line `Verdict: PASS`, `Verdict: CONCERNS`, `Verdict: FAIL`, or `Verdict: BLOCKED`. Do not edit any file other than ./review.md.
