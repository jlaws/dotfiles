---
status: accepted
topic: workflow
created: 2026-11-16
updated: 2026-11-25
---
# Consult commands offer references and delegation

## Context

The consult commands (j-arch, j-cloud, j-devops, j-docs, j-frontend, j-lang, j-data) named their
domain reference library only inside a conditional sentence: delegate to the specialist agent, which
loads it. In the first KB eval baseline (2026-11-16, KB 479cc63) the model declined to delegate in
all 84 kb runs of these commands, and in all 12 of j-system-design. One stored run gave the reason:
"the file is 25 lines, so a direct review costs less than a handoff". Stored graded files cite
`.kb-references/` in only 1 of 12 j-cloud runs, none for the other consult commands with evidence,
and 4 of 12 j-system-design runs. So the library was named only on a path the model never took
(`evals/README.md`, Asset determination).

## Decision Drivers
* **Must reach the domain library** whether the model works inline or delegates
* **Must keep delegation available**: it fired in the pilot (`evals/README.md`, Pilot measurements)
* **Should be checkable**: a list that drifts from its agent is worse than none

## Decision

We will have each consult command carry **its agent's full reference index beside optional
delegation**, and let the model choose. The command says "Load the relevant references for the
topic, based on the diagnostic context:" followed by the agent's `Reference library at` and
`Also see` blocks, names verbatim, in that tree's path form (`~/.agents/references/<category>/`
in `.agents`, `.codex`, and `.gemini`, `~/.claude/references/<category>/` in `.claude/commands` and
`.claude/agents`, and `${CLAUDE_SKILL_DIR}/../../references/<category>/` in `.claude/skills` (listed for completeness of the tree forms);
sync expands the `~` forms to the absolute home path, see
[KB reference paths never resolve against the project](kb-reference-paths-never-resolve-against-the-project.md)); the delegation sentence reads "you may
instead delegate to". j-system-design carries each perspective agent's index the same way. The full
index, not a curated j-observe-style shortlist: it carries everything the agent would load and can
be tested for equality. Every agent index lists its whole directory unless the agent is in
`CURATED_REFERENCE_INDEXES`, so the copies inherit no gaps.

## Rationale

Inline work is what the model chooses for small consults; the list gives that path the library.
Delegation stays for large ones. Copying the index rather than pointing at the agent file costs one
list per command but saves a Read and works across tools, where agent paths differ.

## Consequences
**Gained**: the inline path names the library; a `read-refs` process grader on each consult case
measures whether it is read.
**Accepted**: the lists duplicate the agent indexes, an exception to keeping one source of truth.
`test_consult_commands_list_their_agent_references` fails when any command copy or Gemini agent
index drifts from the `.claude` agent.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| Delete the commands | Each is referenced, so none meets the skill-audit deletion bar | 2026-11-16 |
| Unconditional delegation | A subagent rebuilds context for every consult, and agents pin their own tier (`commands-inherit-the-session-model.md`); small consults do not need it | 2026-11-16 |
| Delegation-only (the old wording) | Declined in all 84 baseline kb runs, so the library was named only on a path never taken | 2026-11-16 |
| Strip delegation | Removes a path the pilot saw fire | 2026-11-16 |
| Leave agent indexes with gaps | Neither the agent nor its commands named files that sat in the directory; the curated agents are listed with a reason instead | 2026-11-25 |
| Point at `agents/<agent>.md` | Costs an extra Read, and agent paths differ by tool and install | 2026-11-16 |

## Enforcement
- Owner: `.claude/commands/j-{arch,cloud,devops,docs,frontend,lang,data,system-design}.md` and their
  `.agents/`, `.codex/`, `.gemini/` twins
- Pinned by: `tests/test_agent_config.py::AgentConfigArchitectureTests::test_consult_commands_list_their_agent_references`
- Pinned by: `tests/test_agent_config.py::AgentConfigArchitectureTests::test_agent_reference_indexes_cover_their_directory`
  and `::test_curated_reference_indexes_are_live`

## Reversal Conditions
Across a later baseline, inline `read-refs` passes in under 1 of 6 kb runs, pooled over both models, for at
least 4 of the 8 commands, or a consult case's kb score drops at least 1/3 on both models.

## Related
- [Commands inherit the session model](commands-inherit-the-session-model.md)
- [KB evals run on a staged plugin](kb-evals-run-on-a-staged-plugin.md)
- [Eval README](../../../evals/README.md)
