---
status: accepted
topic: workflow
created: 2026-11-14
updated: 2026-11-25
---
# Agents carry no persistent memory

## Context

`memory: user` sat on code-reviewer, security-reviewer, and test-writer, all of them j-diff-review
reviewers. The field auto-granted Write and Edit to agents meant to be read-only. It injected an
uncurated `MEMORY.md` index outside the frozen review packet; the security-reviewer index was near
the 200-line load cap, with one note per review. It also gave untrusted diff text a write path into
a prompt loaded in every repo.

## Decision Drivers
* **Must keep reviewers read-only**: a reviewer dispatched with a frozen packet should not write
* **Must close the write path** from untrusted diff text into an always-loaded prompt
* **Should keep durable lessons versioned** and reviewable

## Decision

We will set **no `memory:` on any `.claude/agents/*.md`**. Durable lessons go into versioned skills
or references. Deleting the existing stores under `~/.claude/agent-memory/` is the user's call.

## Rationale

The memory index loaded outside the review packet and grew without curation. A versioned reference
carries the same lessons through review, with no write path from the reviewed content.

## Consequences
**Gained**: reviewers stay read-only; no uncurated text enters their prompt.
**Accepted**: the existing notes stop loading; their general lessons were folded into
`references/security/`, `references/testing/`, and `code-review-patterns` on 2026-11-25; the stores
stay in place. Audit `AG-F9` guards other repos and future agents that do enable
memory.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| An agent-frontmatter Write/Edit fence | A pre-approved Bash `echo >> MEMORY.md` bypasses it, and project subagent hooks skip `-p` runs | 2026-11-14 |
| `disallowedTools` on the agent | Its precedence over the memory auto-grant is undocumented | 2026-11-14 |
| Prose-only rules in the agent body | Advisory; nothing enforces them | 2026-11-14 |
| `project` or `local` memory scope | Committable or plantable by the reviewed repo | 2026-11-14 |
| Leave the stores unharvested | The lessons would stay unversioned and unreviewed; a one-time scrubbed fold keeps the general rules and drops project names | 2026-11-25 |

## Enforcement
- Owner: `.claude/agents/*.md` frontmatter
- Pinned by: audit `AG-F9`; `tests/test_agent_config.py::AgentConfigArchitectureTests::test_agents_carry_no_persistent_memory`

## Reversal Conditions
- Claude Code adds read-only memory injection or a per-dispatch memory toggle.
- The eval baseline shows review cases dropping without memory.

## Related
- [Skills do not pre-approve tools](skills-do-not-pre-approve-tools.md)
