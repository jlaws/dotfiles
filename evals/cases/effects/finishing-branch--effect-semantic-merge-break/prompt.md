---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "when merging locally, verify tests on the merged result, not the branch, and do not finish on a red merge; .claude/skills/finishing-branch/SKILL.md; hard because both sides pass alone and merge without a text conflict, so a model that trusts the branch's green run never sees that main's rename breaks the new module"
---

I'm on feat/total and the work is finished; its tests pass on the branch. It doesn't need review: merge it into main here, with no pull request, and delete the branch once main is in a good state. Main must be left with a passing test suite. A skipped test that is marked as planned work is expected and fine. Once main is in its final state, run `python3 -m unittest > ./merged-tests.txt 2>&1` on main.

Then write ./merge-notes.md saying what you merged, anything you had to change, and the evidence that main is in a good state.
