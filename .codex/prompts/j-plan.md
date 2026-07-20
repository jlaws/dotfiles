---
name: j-plan
description: "Create a detailed implementation plan from a spec or feature — bite-sized TDD tasks, exact file paths, verification commands. Use when you have requirements and need a plan before coding. Do NOT use for executing an existing plan (use $cmd-j-execute-plan) or exploring what to build (use $cmd-j-brainstorm)."
argument-hint: "<spec or feature description>"
---

Invoke the `writing-plans` skill before doing anything else. Use it to turn the spec below into an implementation plan.

First gather context: detect the repo shape (entry points, key modules, test runner, config) so the plan can name exact paths and commands. Then write the plan following the skill — bite-sized (2-5 min) TDD tasks, complete code, exact commands with expected output, no placeholders. Save it to `docs/plans/YYYY-MM-DD-<feature-name>.md`.

Finish with the skill's Execution Handoff: present the execution options (inline via `$cmd-j-execute-plan`, subagents, new session, or manual).

Spec: $ARGUMENTS

If no arguments provided, ask for the spec or point to the design doc from `$cmd-j-brainstorm`.
