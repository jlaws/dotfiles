---
description: "use the gap vocabulary by its meaning: UNPROVEN for a criterion mapped but never demonstrated by a run, UNVERIFIED for an unconfirmed external claim, UNKNOWN for an open question; .claude/skills/analysis-output-patterns/SKILL.md; hard because without the definitions a model reads UNVERIFIED as not yet verified and applies it to the drill that never ran"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Leadership wants a launch-readiness readout for checkout v2. The launch criteria and all the evidence we have are in ./readiness/. Write ./readiness.md with a Summary of at most 3 bullets that leads with the finding, a table with one row per criterion saying whether the evidence shows it is met, and a Caveats section. Mark every gap with exactly one of our gap markers, UNKNOWN, UNVERIFIED, or UNPROVEN, choosing the marker whose meaning fits that gap. State the sample size and time window behind every aggregate figure, and use only figures that appear in the readiness files.
