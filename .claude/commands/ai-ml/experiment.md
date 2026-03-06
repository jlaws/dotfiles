---
name: experiment
description: "Create and run a timestamped experiment script with self-contained folder. Use when running ML experiments, parameter sweeps, or evaluation studies."
argument-hint: <experiment-description>
---

Before creating the experiment, gather diagnostic context:

1. **Detect experiment mode** from $ARGUMENTS:
   - "re-run" / "resume" → locate existing experiment folder, re-run its `experiment.py`, update report
   - "series" / "sequence" → sequential/adaptive multi-experiment workflow
   - Otherwise → single experiment (default)

2. **Check project structure** — scan for `pyproject.toml`, existing `experiments/` directory, available datasets (`game_data/`), and model checkpoints.

3. **Review recent experiments** — `ls -dt experiments/*/` (latest 5) to avoid duplicating work and to inform design.

4. **Check available infrastructure** — look for `infra/ray_cluster.py` to determine if distributed submission is available.

Read references/ai-ml/experiment-runner.md and follow its workflow to create and run: $ARGUMENTS
