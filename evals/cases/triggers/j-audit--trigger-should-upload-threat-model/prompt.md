---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

We are about to ship a file-upload endpoint that accepts images from logged-in users and stores them in S3. Before launch I want a proper threat model of it: STRIDE categories, risk scores, and a mapping of the mitigations to SOC 2 controls.
