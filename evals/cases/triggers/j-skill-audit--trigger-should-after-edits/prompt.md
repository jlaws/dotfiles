---
description: "a user who just edited and renamed several knowledge-base assets wants the whole tree checked for conformance; .claude/commands/j-skill-audit.md; should"
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

I just edited several commands and two agents under .claude/ and renamed a skill. Check that the whole knowledge base is still conformant: frontmatter, cross-references, whether the docs still match, anything that points at the old skill name.
