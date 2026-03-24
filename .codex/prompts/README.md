# Codex Slash Prompts

These markdown files are the Codex slash-command registry.

- Codex discovers custom slash completions from `~/.codex/prompts/`.
- This repo keeps the active prompt files in `.codex/prompts/`.
- Filenames are flattened as `{category}-{name}.md` because Codex prompt discovery is flat.
- The older `.codex/commands/` tree remains as the source-organized library that mirrors the Claude command layout.

After syncing this repo into `~/.codex/`, these prompts should appear in Codex slash completion.
