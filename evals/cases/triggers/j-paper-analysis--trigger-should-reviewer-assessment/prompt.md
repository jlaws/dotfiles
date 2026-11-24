---
description: "a conference reviewer wants a three-pass reading, critical evaluation, and scored recommendation; .claude/commands/j-paper-analysis.md; should"
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

I am a reviewer on an ICLR submission and need a rigorous read of it: a three-pass reading, a critical evaluation, and a reviewer-style score with a recommendation. The abstract: "We propose RouteKV, a routed KV-cache compression scheme for decoder-only language models. RouteKV halves decoding latency at 32k context while matching full-cache perplexity on a held-out corpus." Do the full analysis from what is here.
