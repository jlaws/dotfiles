# Task: Confirm Codex Treats a Model-Written $skill as Explicit

## Context
Found while planning `feat/harness-best-practices`, which turned off Codex implicit invocation for four user-only command skills.

## Problem
`.agents/skills/executing-plans/SKILL.md:82` tells Codex to run `$cmd-j-diff-review`. If Codex treats a `$skill` mention written by the model as implicit invocation, turning implicit invocation off for `cmd-j-diff-review` would break the plan-close chain.

## Scope
1. Check the Codex docs, or run a local Codex session, to confirm how a model-written `$skill` is classified.
2. Record the answer in `docs/adr/workflow/side-effect-commands-are-user-invoked.md` before ever adding `agents/openai.yaml` to `cmd-j-diff-review`.
