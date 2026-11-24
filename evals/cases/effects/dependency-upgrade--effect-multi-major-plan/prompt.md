---
description: "step one major version at a time in the order runtime, framework, data layer, then test tooling, each on its own branch with a tag, after a recorded baseline and the migration notes for every intermediate major; .claude/skills/dependency-upgrade/SKILL.md; hard because jumping straight to the target and upgrading test tooling first both look efficient"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Our Node service is behind. Current to target versions: Node 12 to 22, express 3.21 to 5.1, mongoose 5.13 to 8.x, jest 26 to 29. It has about 70% test coverage and deploys weekly. You cannot install anything here and there is no repository to inspect, so produce the plan only.

Save an executable upgrade plan to ./upgrade-plan.md. It must give the order of the steps, what each step upgrades and its version hop, how each step is isolated and how we get back to the state before it, what we record before the first step as a baseline, and what we read and check before and after each step. Keep the plan under 150 lines: one short entry per step.
