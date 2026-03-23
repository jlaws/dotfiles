# Codex Prompt Library

Codex does not currently auto-discover a custom `commands/` folder the way some other agents do.

This directory is still useful as a maintained prompt library:

- `commands/` contains reusable task prompts and workflows.
- `references/` contains longer background material those prompts point to.
- `agents/` contains Codex-native subagent definitions that Codex can actually load.
- `~/.agents/skills/` contains Codex-native skills that Codex can auto-select or load explicitly.

Treat these markdown command files as reusable templates, not as auto-registered slash commands.
