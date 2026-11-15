---
type: llm
focus: { source: file, path: broken-skill/SKILL.md }
---

All claims are required.
PASS if the frontmatter still has name: broken-skill and now has a description field.
PASS if the description states only triggering conditions (when to use the skill, for example when asked to summarize a changelog or release notes) and does not describe the steps or workflow of the skill (such as "three bullets" or "summarizes into").
PASS if the description is a single short line under 200 characters.
FAIL if the description summarizes what the skill does step by step instead of when to use it.
