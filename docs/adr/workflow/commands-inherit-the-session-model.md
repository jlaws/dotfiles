---
status: accepted
topic: workflow
created: 2026-10-02
updated: 2026-10-02
---
# Commands inherit the session model

## Context

Every `.claude/commands/*.md` except `j-next` pinned `model: opus` or `model: sonnet`. Claude Code
treats a command whose `model` differs from the session's as a model switch: "the next request
reads the entire conversation history with no cache hits. The session model resumes on your next
prompt" (code.claude.com/docs/en/prompt-caching, 2026-10-02). Whenever a pin differed from the
session model, invoking the command mid-session paid that miss. The pin also covered only the
command's first turn: in an interactive command (`j-brainstorm`, `j-plan`, `j-new`), every later
answer ran on the session model anyway. `j-new` already told authors to leave `model` unset, and the
commands contradicted it. Anthropic's cost-and-intelligence guide ranks caching as the largest cost
lever.

Subagents differ. A subagent resolves its model at spawn (per-call `model`, then agent frontmatter,
then `CLAUDE_CODE_SUBAGENT_MODEL`, then the main model) and builds its own cache, so an agent's
pinned tier costs the parent nothing (code.claude.com/docs/en/sub-agents). Effort differs too: on
Opus 5.5, Sonnet 5.5, and Fable 5.1 with an API key or a subscription, changing effort keeps the
cache.

## Decision

Commands never set `model`; they inherit the session model. A tier pin belongs only on an asset that
starts a fresh context: agents (required by `AG-F8`) and `context: fork` skills. Commands keep
`effort`. Audit check `CM-F7` warns on a command that sets `model`, and `make test` fails on the
warning. Inline skills follow the same rule through `j-new` and `writing-skills` guidance; no skill
sets `model` today.

## Consequences

**Gained**: invoking a command mid-session forces no cache miss. The session alone picks the
parent's model. `j-new` and the commands agree.
**Accepted**: commands that were pinned to `sonnet` for unattended work now run on the session
model. To route them cheaper, start the session with `claude --model sonnet`, or delegate the bulk
to a sonnet-pinned agent. Commands that were pinned to `opus` for design work run on whatever the
session uses, including Sonnet. The `effort` values were chosen alongside the old pins and now apply
to the session model.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Keep the pins | Each mid-session invocation re-reads the whole history uncached, and the pin covers only the first turn of an interactive command | 2026-10-02 |
| Allowlist single-turn commands usually run in a fresh session | "Usually fresh" cannot be enforced, and the list decays | 2026-10-02 |
| Pin every command to the current session model | Breaks the day the session model changes, and repeats one choice in every command | 2026-10-02 |

## Reversal Conditions

- Claude Code keeps the cache across a frontmatter model switch, or holds a command's model for its whole dialogue.
- Measured spend shows session-level routing is too coarse for the unattended commands.
