---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "an accessible tab widget uses tablist, tab, and tabpanel roles with aria-selected, aria-controls, and aria-labelledby, moves between tabs with the arrow keys, and jumps to the first and last tab with Home and End; .claude/references/frontend/accessibility-testing.md; hard because hand-rolled tab widgets usually stop at the arrow keys and leave out Home and End"
---

/kb:j-frontend make the tab component in ./src/Tabs.tsx usable by keyboard and screen-reader users; ./src/AccountPage.tsx shows how it is used. Keep the component's props unchanged and edit ./src/Tabs.tsx in place.
