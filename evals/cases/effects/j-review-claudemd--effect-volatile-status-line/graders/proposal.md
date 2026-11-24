---
type: llm
focus: { source: file, path: claudemd-proposal.md }
---

All claims are required.
PASS if the proposal recommends removing the status line at CLAUDE.md:3 (sprint, open bugs, last deploy) because its content is volatile (it changes week to week, goes stale, or breaks the prompt cache).
FAIL if the proposal recommends updating that line to sprint 49, 9 open bugs, a 11-16 deploy, or any other newer figures.
PASS if the proposal flags the just test rule as violated in session 2, an enforcement gap since the rule already exists.
FAIL if the proposal recommends deleting the src/api Result convention outright; moving it to a module-level CLAUDE.md is allowed.
