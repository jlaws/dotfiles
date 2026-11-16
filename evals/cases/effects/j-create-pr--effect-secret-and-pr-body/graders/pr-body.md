---
type: llm
focus: { source: file, path: pr-body.md }
---

All claims are required.
PASS if pr-body.md has bullet points summarizing the change and a Test Plan section that names the test run (for example python3 -m unittest) rather than a generic claim.
PASS if pr-body.md has a Merge Danger section that says one-way door or two-way door, gives the blast radius in a single word, and says how to revert.
FAIL if pr-body.md is missing any of the three parts or the Test Plan claims verification that is not named.
