---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

We shipped release v0.2 of this tool. The release notes say:
- Renamed the --out flag to --output.
- Added a --delimiter flag for TSV input. (This was merged and then reverted before the release was cut, so it is not in the shipped code.)

Bring the documentation in line with what actually shipped, by checking the code. Make only the changes the shipped code requires. Do not commit anything. Save ./sync-summary.md listing each doc change you made and each release-note item you did not document, with the reason.
