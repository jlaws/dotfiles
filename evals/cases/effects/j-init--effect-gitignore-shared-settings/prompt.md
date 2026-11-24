---
description: "the .gitignore covers .claude/settings.local.json and the stack's secret-bearing files, while .claude/settings.json stays committable because it is the project file meant to be shared; .claude/commands/j-init.md; hard because a stock Python template never names the personal settings.local.json, and a model that treats .claude/ as local tool state ignores the whole directory, shared settings included"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-init Finish setting up this repository, which collaborators will clone. Its CLAUDE.md, AGENTS.md, GEMINI.md, and project settings in .claude/settings.json are current, so leave them alone. Write the .gitignore it is missing at ./.gitignore, so that git keeps out everything that should not be shared with everyone who clones the repo and keeps everything that should. Do not install anything and write no git state.
