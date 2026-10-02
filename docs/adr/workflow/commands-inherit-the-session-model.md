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
commands contradicted it. Anthropic's [cost-and-intelligence guide](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)
(fetched 2026-10-02) ranks caching as the largest cost lever.

Subagents differ. A subagent resolves its model at spawn (per-call `model`, then agent frontmatter,
then `CLAUDE_CODE_SUBAGENT_MODEL`, then the main model) and builds its own cache, so an agent's
pinned tier costs the parent nothing (code.claude.com/docs/en/sub-agents). Effort differs too: on
Opus 5.5, Sonnet 5.5, and Fable 5.1 with an API key or a subscription, changing effort keeps the
cache; on Bedrock, Vertex, and gateways it re-reads the history.

## Decision

Commands never set `model`; they inherit the session model. A tier pin belongs only on an asset that
starts a fresh context: agents (required by `AG-F8`) and `context: fork` skills. Audit check
`CM-F7` warns on a command that sets `model`, and `make test` fails on the warning. Inline skills
follow the same rule through `j-new` and `writing-skills` guidance; no skill sets `model` today.

Commands also inherit the session's effort. A command sets `effort` only to lower it (`low`,
`medium`) for cheap, procedural work; it never raises it. In the guide's measurements `xhigh` cost
2.5x `high` for 1.4 points on long-horizon coding (Opus 5.5 on SWE-bench Pro), and research and
knowledge work gained nothing measurable above `medium` (Fable 5). On first-party models the lowered
tiers cost no cache, per the effort rule above.

## Consequences

**Gained**: invoking a command mid-session forces no cache miss. The session alone picks the
parent's model. `j-new` and the commands agree.
**Accepted**: commands that were pinned to `sonnet` for unattended work now run on the session
model. To route them cheaper, start the session with `claude --model sonnet`, or delegate the bulk
to a sonnet-pinned agent. Commands that were pinned to `opus` for design work run on whatever the
session uses, including Sonnet. Commands that were pinned to `high` or `xhigh` (design, debugging,
review, and audit commands) now run at the session's effort; raise it for the session with `/effort`
when a task needs more. That includes the inline passes of `j-audit` and `j-config-audit`; their
deep review still goes to `security-reviewer`, an agent pinned to `opus`, and the config audit's
report now carries a `Scanned:` coverage line and a BLOCKED verdict so a shallow run shows.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Keep the pins | Each mid-session invocation re-reads the whole history uncached, and the pin covers only the first turn of an interactive command | 2026-10-02 |
| Allowlist single-turn commands usually run in a fresh session | "Usually fresh" cannot be enforced, and the list decays | 2026-10-02 |
| Pin every command to the current session model | Breaks the day the session model changes, and repeats one choice in every command | 2026-10-02 |
| Keep `high` and `xhigh` effort pins on demanding commands | Set by judgment and never measured; the measured `xhigh` gain was 1.4 points for 2.5x the cost of `high`, and `/effort` raises the whole session when a task needs it | 2026-10-02 |

## Enforcement
- Owner: `.claude/commands/*.md` frontmatter
- Pinned by: `tests/test_reference_tree.py::CommandModelTests` and `::ReferenceTreeTests::test_no_warning_findings` (`CM-F7`, no `model`); `tests/test_agent_config.py::AgentConfigArchitectureTests::test_claude_command_effort_only_lowers_the_session_level` (effort `low` or `medium` only). Inline skills: nothing, convention only (`j-new`, `writing-skills`)

## Reversal Conditions

- Claude Code keeps the cache across a frontmatter model switch, or holds a command's model for its whole dialogue.
- Measured spend shows session-level routing is too coarse for the unattended commands.

## Related
- [Merge agent configs during setup](merge-agent-configs-during-setup.md)
