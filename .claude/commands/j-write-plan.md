---
name: j-write-plan
description: "Create a structured implementation plan with bite-sized tasks, exact file paths, and TDD integration. Use when you have requirements and need a detailed plan before coding. Do NOT use for changing existing plans (edit the file directly)."
argument-hint: "<spec, design doc path, or feature description>"
---

Invoke the `writing-plans` skill via the Skill tool before doing anything else. Use it to produce the plan for the input below. Save to `docs/plans/YYYY-MM-DD-<feature>.md`.

Input: $ARGUMENTS

If no arguments provided, ask what needs to be planned or check for a recent design doc in `docs/plans/`.
