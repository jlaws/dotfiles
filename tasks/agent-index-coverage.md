# Task: Make agent reference indexes cover their directories

## Context
Consult commands now copy their agent's reference index verbatim
(`docs/adr/workflow/consult-commands-offer-references-and-delegation.md`), and
`test_consult_commands_list_their_agent_references` checks command against agent only.

## Problem
Agent indexes omit references that exist in their directory, so neither the agent nor the commands
name them: `.claude/agents/architecture-specialist.md` omits `architecture/deployment` and
`architecture/error-handling-patterns`; `.claude/agents/language-specialist.md` omits
`languages/concurrency-patterns`. Nothing checks an index against `.claude/references/<dir>/`.

## Scope
1. Add the missing names to the agents (all trees) and the consult commands, or record why each is left out.
2. Add a test that each agent's `Reference library at <dir>` block lists every file in that directory,
   with an explicit exception list.
