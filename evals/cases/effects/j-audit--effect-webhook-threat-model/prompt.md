---
description: "threat-model the design per STRIDE with an impact x likelihood score per threat and a compliance mapping of the mitigations; .claude/commands/j-audit.md; hard because the body is HMAC-signed, so the design reads as authenticated and the replay of a validly signed event is easy to miss"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-audit threat-model the payment webhook receiver described in ./docs/webhook-design.md. Save the threat model to ./threat-model.md with one section per STRIDE category (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege). Each section lists its threats with a risk score as impact x likelihood (each 1 to 4) and the mitigation that would address it. End with a table mapping the mitigations to PCI DSS or SOC 2 controls.
