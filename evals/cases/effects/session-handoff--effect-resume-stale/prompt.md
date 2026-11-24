---
description: "on resume, check the handoff's Files Modified claims against the files and rerun its tests before trusting it; .claude/skills/session-handoff/SKILL.md; hard because the handoff says done and all green, and a model tends to start from Next Steps"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

A previous session left the handoff below, and I want you to resume from it. Do the work it describes and save a short note to ./resume.md stating what you verified before trusting the handoff, what you found, and what you did.

--- handoff ---
# Handoff: Fix empty mean
Status: Ready for Review
## Files Modified
| File | Change | Status |
|------|--------|--------|
| calc/ops.py | added guard so mean([]) returns 0.0 | done |
| tests/test_empty.py | test for mean([]) | done |
## Test Results
- Last run: python3 -m unittest -> pass, all green
## Next Steps
1. Update README to mention that the mean of an empty list is 0.0
---
