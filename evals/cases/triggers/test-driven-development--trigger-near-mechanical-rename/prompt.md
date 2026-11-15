---
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Rename the helper parseCfg to parseConfig across the repo. The existing tests already cover every call site, so I only need the rename done and the suite confirmed green.
