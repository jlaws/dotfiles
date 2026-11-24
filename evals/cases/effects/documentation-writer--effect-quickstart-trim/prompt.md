---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "keep a quickstart to 3 to 5 steps, state the concrete outcome up front, and make every code block runnable as-is with no elisions; .claude/references/documentation/quickstart-template.md; hard because a rewrite of a nine-step draft tends to keep six or seven steps, and the draft's elided snippet reads as normal sample code"
---

Use the kb:documentation-writer agent to rewrite the draft quickstart in ./docs/quickstart.md for developers trying widgetkit for the first time. The public API is in ./widgetkit/client.py, and ./docs/configuration.md and ./docs/webhooks.md already cover configuration and webhooks. Overwrite ./docs/quickstart.md with the result. Decide how many steps the quickstart has and what to leave out of it, and point readers to where the material you leave out is covered.
