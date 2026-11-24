---
description: "an agent that enables memory uses user scope, lists Read, Write, and Edit, and its body says what may be saved, when memory is read and written, and that memory never suppresses a finding; .claude/skills/code-agent-meta-patterns/SKILL.md; hard because the memory body rules are house conventions a model would not volunteer, and untrusted diffs tempt saving their text"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Design a Claude Code subagent definition for a code reviewer and save it to ./agents/reviewer.md. It reviews pull request diffs from untrusted contributors in a public repository and should get better over time by remembering review techniques between runs. Use valid agent frontmatter (name, description, tools, and whatever else the memory feature requires); the tools list must include Read, Write, and Edit so the agent can maintain its memory files and write the body so that what the agent may and may not save to memory is spelled out, when in its run it reads and writes memory, and whether memory can ever change a finding. Memory must be stored per user, not inside the repository.
