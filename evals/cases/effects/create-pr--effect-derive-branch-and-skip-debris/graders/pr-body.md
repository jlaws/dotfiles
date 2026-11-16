---
type: llm
focus: { source: file, path: pr-body.md }
---

All claims are required.
PASS if pr-body.md has bullet points of what changed, a Test Plan naming the verification that actually ran (such as python3 -m unittest), and a Merge Danger section with one-way or two-way door, a one-word blast radius, and how to revert.
FAIL if any of the three parts is missing or the Test Plan is generic.
