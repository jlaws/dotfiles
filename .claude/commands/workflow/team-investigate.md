---
description: "Competing hypothesis debugging — multiple agents investigate different theories in parallel."
---

Before invoking the skill, gather evidence:

1. **Check git status** for current state.
2. **Check recent commits** (`git log --oneline -20`) for potential culprits.
3. **Gather error context**: If $ARGUMENTS mentions a test, run it to capture output; if it mentions logs, search for relevant log files.
4. **Identify affected components/files** from the error description.

Then orchestrate the investigation team:

1. **Create a team** with TeamCreate.
2. **Enter delegate mode** (Shift+Tab) — lead coordinates only, does not edit files.
3. **Form 3-4 competing hypotheses** about the root cause based on the gathered evidence.
4. **Create tasks** with TaskCreate — one per hypothesis, including:
   - The hypothesis to test
   - The error context and relevant file paths
   - File ownership: **read-only for all investigators** (no edits — investigation only)
   - Instructions to search for evidence supporting OR refuting their hypothesis
5. **Spawn one investigator per hypothesis** (Explore type) into the team, assign tasks via TaskUpdate.
6. **Adversarial challenge**: after initial findings, instruct investigators to review each other's conclusions via SendMessage. Each should attempt to poke holes in other hypotheses — strengthen or eliminate theories through debate.
7. **Collect findings** from TaskList and messages, rank by evidence strength.
8. **Present unified investigation report** with:
   - Most likely root cause (with evidence)
   - Alternative explanations considered (with reasons for lower confidence)
   - Points of agreement/disagreement between investigators
   - Recommended next steps
9. **Shut down team** after report is assembled.

Invoke the testing:debugging-methodology skill for investigation methodology: $ARGUMENTS
