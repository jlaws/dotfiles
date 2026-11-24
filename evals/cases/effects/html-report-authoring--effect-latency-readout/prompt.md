---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "a findings report leads with the BLUF headline, labels measured versus inferred data with the gap stated, and collapses raw data in native details; .claude/skills/html-report-authoring/SKILL.md; hard because models open a report with a title and a methodology line before the finding, and blur the unmeasured region into the results"
---

Write ./readout.html, a single self-contained HTML file (no network, works offline) presenting this latency experiment to engineers who want to act on it.

Data (p95 latency in ms, before the cache change then after): us-east 120 then 70, eu-west 180 then 110, ap-south not measured after the change (the test node was down). Traffic was synthetic, replayed from last Tuesday. We believe the ap-south result will be similar to the others because its cache hit ratio was the same before the change, but we have no direct measurement.

The readout must lead with the headline result, clearly separate what was measured from what is inferred and state what is missing, say what traffic the numbers were measured under, keep the raw numbers collapsed until the reader opens them using native HTML elements and no framework, and include a button that copies the findings as Markdown.
