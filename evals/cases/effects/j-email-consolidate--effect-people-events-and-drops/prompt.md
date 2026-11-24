---
description: "route People and Events into the four ordered subsections, keep AI cultural moments, and apply the always-drop list for sponsors, indie roundups, off-topic content, and see-also links; .claude/commands/j-email-consolidate.md; hard because the robot monk ceremony reads as off-topic culture a model would drop, while the lactase podcast and the encyclopedia link read as content to keep"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-email-consolidate merge the per-email reports in ./emails into one weekly digest at ./digest.md. Requirements: Research Papers is a markdown table with the columns Paper, Authors, URL, Note, in that order, and a paper whose PDF download failed stays there with a note saying it needs a manual download and why. Tools & Repos is a markdown table with the columns Name and Description. People & Events is grouped under ### subheadings in this order: Podcasts & long-form interviews; Funding, IPOs, talent; Conferences & workshops; Cultural moments. Keep AI and robotics cultural appearances and viral demos under Cultural moments. Drop sponsor placements, indie product-roundup entries with no release or funding context, podcast or article content with no AI angle, and encyclopedia-style see-also links. Keep every kept item's external link verbatim, and leave no leftover HTML comment markers in the file.
