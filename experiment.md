---
description: Create and run a new timestamped experiment script, saving results to self-contained folder
argument-hint: <experiment-description>
---

# Experiment Runner

Create and run a single-file experiment script based on the user's description: $ARGUMENTS

## Experiment Folder Structure

Each experiment is self-contained in its own folder:
```
experiments/<datetime>-<slug>/
    experiment.py      # Main experiment script
    report.md          # Markdown report with findings
    game_data/*.npz    # GameRecord played and saved
    data/*.csv         # Result data files
    figures/*.png      # Generated figures
    checkpoints/*.pt   # Model checkpoints (if any)
```

## Modes

- **Single experiment** (default): Create one experiment folder, run it, write report
- **Re-run/resume**: If user indicates "re-running" or "resuming", re-run existing experiment.py and update report.md
- **Sequential experiments**: If user mentions running a "series" or "sequence" of N experiments, use the adaptive experimentation workflow (see below)

## Steps (Single Experiment)

1. **Generate timestamp and slug**: Use today's datetime formatted as `%Y-%m-%d_%H-%M` and create a kebab-case slug from the experiment description (e.g., "2025-12-30_00-27-onestep-q-estimation")

2. **Create experiment folder and script**:
   - Create the experiment folder: `experiments/<datetime>-<slug>/`
   - Write the main script at `experiments/<datetime>-<slug>/experiment.py`
   - The script should save outputs to its own folder (data/, figures/, checkpoints/)
   - The experiment script should report how long it took to execute.

3. **Run the experiment to generate data**: Execute the script with `uv run experiments/<datetime>-<slug>/experiment.py`

For long running experiments, save intermediate results like game_data so that we can still analyze data if not all iterations complete and we stop the experiment earlyl.

4. **Create an analysis script to analyze the data**: If needed, write additional analysis scripts in the same experiment folder.

5. **Complete the report skeleton**: The experiment script generates a report skeleton via `write_report_skeleton()`. Open `experiments/<datetime>-<slug>/report.md` and fill in:
   - User prompt
   - Methodology section and any instructions that might help when resuming
   - Analysis of results
   - Key findings (bullet points)
   - Conclusions and next steps
   - How long the experiment took to run

6. **Git commit**: Stage and commit the experiment script and report (not data, figures, checkpoints):
   ```bash
   git add experiments/<datetime>-<slug>/experiment.py experiments/<datetime>-<slug>/report.md
   git commit -m "Add experiment: <experiment-slug>"
   ```

   Note: The report automatically captures the git commit hash at generation time via `_get_git_commit()` in `write_report_skeleton()`.

7. If there was any difficulty that you encountered in writing working code, comment on what changes to the codebase or Claude skill files would benefit you in the future.

## Sequential/Adaptive Experiments

When user requests a "series" or "sequence" of N experiments (e.g., "run 10 experiments to tune MCTS parameters"), use this adaptive workflow where each experiment is designed after analyzing prior results.

### Sequential Folder Structure

```
experiments/<datetime>-<research-slug>/
    README.md                              # Overview and research goal
    <datetime>-<exp-slug>-01/
        experiment.py
        report.md
        data/
        figures/
    <datetime>-<exp-slug>-02/
        experiment.py
        report.md
        data/
        figures/
    ...
    final_report.md                           # Final report synthesizing all findings
```

### Sequential Workflow

1. **Initialize sequence folder**: Create parent folder with timestamp and research slug, write README.md with research goal

2. **For each experiment (1 to N)**:

   **A. Design Phase** (skip for first experiment)
   - Review reports from completed experiments
   - Identify remaining unknowns and uncertainties
   - Design next experiment to address the most important unknown

   **B. Execute Phase**
   - Create subfolder: `<datetime>-<exp-slug>-<NN>/`
   - Write experiment.py with clear hypothesis
   - Run: `uv run experiments/<sequence>/<subfolder>/experiment.py`
   - Write report.md with findings and implications for next experiment
   - Git commit the experiment script and report

   **C. Analyze Phase**
   - Extract learnings from results
   - Update beliefs about the research question
   - Identify next priority (feeds into next iteration's Design Phase)

3. **Create synthesis.md**: After all experiments complete, summarize:
   - Table of all experiments and key findings
   - Overall conclusions and recommended configuration
   - Open questions for future work

### Sequential Experiment Principles

- **Test one variable**: Change only one thing at a time when possible
- **Clear hypothesis**: Know what hypothesis you are testing or confirming, and have a clear expected outcome guessed.
- **Build on prior work**: Each experiment references which earlier experiments informed its design
- **Adapt the plan**: The sequence evolves based on findings - don't stick rigidly to initial plan

### Example Sequence

```
experiments/2026-01-17_12-00-tune-mcts-vs-katago/
    README.md
    2026-01-17_12-00-baseline-params-01/
    2026-01-17_12-00-num-simulations-sweep-02/
    2026-01-17_12-00-cpuct-sweep-03/
    2026-01-17_12-00-combined-best-04/
    ...
    synthesis.md
```

## Datasets

Experiments will often require instantiating datasets for training and validation. Data can be generated as part of a single experiment (e.g. an overfitting test), and re-used by other experiments. The default dataset to use is the list of path expressions in game_data/9x9/dataset.txt

GoDataset now supports multiple directories:

# Single directory (unchanged behavior)
```
dataset = GoDataset("game_data/9x9/dev-train")
```

# Multiple directories (new)
```
dataset = GoDataset([
    "game_data/9x9/dataset1",
    "game_data/9x9/dataset2",
    "game_data/9x9/dataset3",
])
```

## Modifying code and training models

Experiments will often require training multiple runs of models. Feel free to fork alpha_go/train.py entirely to the new experiment file. Model architecture modifications should be inlined in the experiment script. When modifying code, you may only create and edit new *.py files in experiments/ subdirectory. If you find that it would be better to edit files in src/, please ask the user for permission first.

## Playing Go Games and generating data

All experiments should use `alpha_go.gameplay.play_game` or `uv run alpha_go.self_play` to evaluate and generate data. If these functions are not able to serve the experiment, suggest changes to these functions.

## Eval Logging

All game evaluations should log to the SQLite database for tracking:

```python
from alpha_go.gameplay import play_game

# Log games to database automatically
record = play_game(black_agent, white_agent, log_to_db=True)
```

View eval results in the dashboard:
```bash
uv run -m alpha_go.eval_dashboard --port 8090
```

Query results programmatically:
```python
from alpha_go.eval_log import EvalLogDB

db = EvalLogDB.get_instance()
win_rate = db.get_win_rate_vs_baseline(
    checkpoint="checkpoints/step_50000.pt",
    baseline_agent="gnugo5",
    as_black=True,
)
```

## Running multiple experiments in parallel

We may want to run experiments on remote machines to parallelize research. The workflow for running experiments across multiple machines looks like:

1. Spin up the ray cluster if it is not already up: `./infra/ray_cluster.py up`
2. Submit experiments using the ray_cluster.py submit command:

```bash
# Basic submission (runs on any worker)
./infra/ray_cluster.py submit -- /root/.local/bin/uv run experiments/<datetime>-<slug>/experiment.py

# With GPU requirement
./infra/ray_cluster.py submit --entrypoint-num-gpus 1 -- /root/.local/bin/uv run experiments/<datetime>-<slug>/experiment.py

# Non-blocking (returns immediately)
./infra/ray_cluster.py submit --no-wait -- /root/.local/bin/uv run experiments/<datetime>-<slug>/experiment.py

# With custom resources
./infra/ray_cluster.py submit --entrypoint-num-gpus 1 --entrypoint-num-cpus 4 -- /root/.local/bin/uv run experiments/<datetime>-<slug>/experiment.py --arg1 val1
```

### Container vs bare metal workers

The Ray cluster supports two worker types with different capabilities:
- **Bare metal** (`bare_metal_worker` resource): Fast to set up, has Python/Ray/uv. Good for training, pure-Python experiments.
- **Container** (`container_worker` resource): Runs in Docker, has `alpha_go_cpp`, KataGo binaries, and all compiled dependencies pre-built.

**Jobs that use `alpha_go_cpp`, KataGo agents, or any compiled C++ binaries MUST be scheduled on container workers.** Use the `--entrypoint-resources` flag to request the `container_worker` resource:

```bash
# Schedule on a container worker (required for alpha_go_cpp / KataGo / compiled agents)
./infra/ray_cluster.py submit --entrypoint-resources '{"container_worker": 1}' -- \
    uv run experiments/<datetime>-<slug>/experiment.py

# Container worker + GPU
./infra/ray_cluster.py submit --entrypoint-resources '{"container_worker": 1}' --entrypoint-num-gpus 1 -- \
    uv run experiments/<datetime>-<slug>/experiment.py
```

For programmatic submission in `03_submit_all.py` scripts, include the resource request:

```python
# Eval jobs using alpha_go_cpp or KataGo need container workers
resources = '{"container_worker": 1}'
cmd = [
    "./infra/ray_cluster.py", "submit", "--no-wait",
    "--entrypoint-resources", resources,
    "--entrypoint-num-gpus", "0.1",
    "--", "uv", "run", script_path, *args,
]
```

Training-only jobs (no compiled deps) can run on any worker and don't need the resource flag.

Import `alpha_go_cpp` directly (pre-built in the dev container and on container Ray workers):

```python
import alpha_go_cpp
```

## Handling data dependencies for remote jobs

Uploading dependencies and downloading dependencies can be done with R2Path objects.

```
# Relative path -> downloads to $ALPHAGO_BASE_DIR/<path>
ckpt = R2Path("checkpoints/step_50.pt")
# Creates: ~/checkpoints/step_50.pt from r2://ttl31d/checkpoints/step_50.pt

# Auto-upload to R2 in background (for distributed launch)
ckpt = R2Path("checkpoints/step_50.pt", upload=True)
# Uploads to R2 in background if local and not already in R2

# Wait for upload to complete before launching workers
ckpt.wait_for_upload()

# Use with open() and Path() directly
with open(ckpt, "rb") as f:
    data = f.read()
```

When uploading a dataset of .npz files, consider zipping before uploading to minimize the number of API calls.

Always check that files exist on R2 before launching the experiment to ray head. You are permitted to create an upload_dependencies.py script in the experiment folder that instantiates R2Path objects and trigger uploads before submitting the actual experiment script to ray.

On Ray workers, `ALPHAGO_BASE_DIR=/root/alphago` and experiments write to `/root/alphago/experiments/<datetime>-<slug>/`.

When running multiple concurrent processes, write stdout / stderr to separate text logs in `text_logs` subdirectory.

## Parallel Ray Eval Jobs

For experiments requiring many parallel evaluations (e.g., evaluating N checkpoints against M opponents), use the multi-script pattern with numbered scripts:

```
experiments/<datetime>-<slug>/
    01_upload_dependencies.py   # Upload checkpoints/models to R2
    02_eval_single.py           # Single eval job (runs on Ray workers)
    03_submit_all.py            # Submit all jobs to Ray
    04_analyze_results.py       # Download results and generate plots
    data/
        job_manifest.json       # Tracks submitted jobs and their params
```

### Workflow

1. **01_upload_dependencies.py**: Upload checkpoints/models to R2 using `R2Path(path, upload=True)`. Save metadata (R2 keys, step numbers) to `data/` for the submit script.

2. **02_eval_single.py**: The worker script that runs on Ray. Takes CLI args (e.g., `--checkpoint`, `--opponent`), downloads from R2 via `R2Path()`, runs eval, saves results to R2. Print a parseable `===RESULT===` block at the end for easy aggregation.

3. **Test one job first**: Before submitting all jobs, verify end-to-end:
   ```bash
   # Eval jobs need container workers for alpha_go_cpp / KataGo
   ./infra/ray_cluster.py submit --entrypoint-resources '{"container_worker": 1}' --entrypoint-num-gpus 0.1 -- \
       uv run experiments/<slug>/02_eval_single.py --checkpoint <one_key>
   ```

4. **03_submit_all.py**: Load metadata from step 1, submit all jobs with `--no-wait`, save job IDs to `data/submitted_jobs.json`. Use small resource requests (e.g., `--entrypoint-num-gpus 0.1 --entrypoint-num-cpus 1`) to maximize parallelism. **Include `--entrypoint-resources '{"container_worker": 1}'`** for eval jobs that use `alpha_go_cpp` or KataGo agents.

5. **04_analyze_results.py**: Download results from R2 with `rclone sync`, parse NPZ/CSV files, compute metrics, generate plots.

### Key Patterns

- Worker scripts should use `R2Path()` to auto-download dependencies
- Upload game data back to R2 at end of each worker job
- Parse job IDs from submit output: `re.search(r"'(raysubmit_[^']+)'", output)`
- Monitor jobs: `ray job list`

See `experiments/2026-01-25_19-24-checkpoint-pair-eval/` for a complete example.


## Distributed Replay Buffer + Trainer + Collector RL

Larger-scale jobs involve parallelizing data collection across multiple nodes and having them communicate with other node services like replay buffers, inference servers, and so forth. Here is an example of how to launch a distributed replay job with the driver code running on the local machine.

```bash
uv run example_recipes/distributed_collect.py
```

The driver script can be run locally, and it launches processes locally or on remote nodes using Ray.

## Output Paths

All outputs are within the experiment folder:
- Script: `experiments/<datetime>-<slug>/experiment.py`
- Data: `experiments/<datetime>-<slug>/data/*.csv`
- Figures: `experiments/<datetime>-<slug>/figures/*.png`
- Report: `experiments/<datetime>-<slug>/report.md`
- Checkpoints: `experiments/<datetime>-<slug>/checkpoints/*.pt`
- Text Logs: `experiments/<datetime>-<slug>/text_logs/*.log`

## Analysis Library

Use `alpha_go.analysis` for standard plots and report generation instead of writing boilerplate from scratch.

### Available Functions

```python
from alpha_go.analysis import (
    # Training data loading
    load_training_run,         # Load CSV + flops JSON for a run
    compute_cumulative_flops,  # Calculate FLOPs from step count

    # Training plots
    plot_loss_vs_flops,        # Training/val loss curves vs FLOPs
    plot_accuracy_vs_flops,    # Policy/value accuracy vs FLOPs
    plot_training_summary,     # Generate all standard plots for a run

    # Go board rendering
    render_board_simple,       # Render board without policy overlay
    render_board_with_policy,  # Render board with policy heatmap
    get_star_points,           # Get star point coordinates for board size
    get_column_labels,         # Get column labels (A-T, skipping I)

    # Reports
    write_report_skeleton,     # Generate markdown skeleton with tables + figures
)
```

### Report Generation

```python
from alpha_go.analysis import write_report_skeleton

write_report_skeleton(
    slug="onestep-q-estimation",
    date_prefix="2025-12-30_00-27",
    experiment_dir=EXPERIMENT_DIR,  # Path to experiment folder
    csv_path=DATA_DIR / "results.csv",
    figures=[FIG_DIR / "results.png"],
    prompt="Test one-step Q estimation vs raw policy",
)
```

### Go Board Visualization

```python
from alpha_go.analysis import render_board_simple, render_board_with_policy
import matplotlib.pyplot as plt
import numpy as np

# Simple board rendering (no policy)
fig, ax = plt.subplots(figsize=(6, 6))
board = np.zeros((9, 9))  # 0=empty, 1=black, 2=white
board[4, 4] = 1  # Black stone at center
render_board_simple(board, ax, title="Game Position", board_size=9)

# Board with policy heatmap overlay
fig, ax = plt.subplots(figsize=(6, 6))
policy_2d = np.random.rand(9, 9)  # Policy probabilities
policy_2d = policy_2d / policy_2d.sum()  # Normalize
render_board_with_policy(
    board=board,
    policy_2d=policy_2d,
    ax=ax,
    legal_moves=[(r, c) for r in range(9) for c in range(9) if board[r, c] == 0],
    chosen_move=(3, 3),  # Highlight chosen move with red X
    title="Policy Distribution",
    board_size=9,
    show_probs=True,      # Show % on high-prob moves
    prob_threshold=0.05,  # Min prob to show label
)
```

### Training Data Loading & Plots

```python
from alpha_go.analysis import load_training_run, compute_cumulative_flops
from alpha_go.analysis import plot_loss_vs_flops, plot_accuracy_vs_flops

# Load training CSV and flops metadata
df, flops_info = load_training_run("my-run", data_dir=str(DATA_DIR))
# flops_info: {"n_params": int, "board_size": int}

# Compute cumulative FLOPs for each step
cumulative_flops = compute_cumulative_flops(df, flops_info)

# Generate loss curves (train batch, train epoch, val)
fig = plot_loss_vs_flops("my-run", data_dir=str(DATA_DIR), output_path=str(FIG_DIR / "loss.png"))

# Policy and value accuracy curves
fig = plot_accuracy_vs_flops("my-run", data_dir=str(DATA_DIR), output_path=str(FIG_DIR / "accuracy.png"))
```

### Expected CSV Schema

Training CSVs should include these columns for compatibility with the plotting library:
- `global_step`: Training step number
- `train_loss`, `val_loss`: Loss values
- `train_eval_loss`: Train loss evaluated at epoch boundaries
- `train_policy_acc`, `val_policy_acc`: Policy head accuracy
- `train_value_acc`, `val_value_acc`: Value head accuracy
- `batch_size` (optional): Defaults to 64 if not present

A companion `<run_name>_flops.json` file should contain:
```json
{"n_params": 1000000, "board_size": 9}
```

## Common Training Patterns

### Cosine Schedule with Warmup

```python
import math
import torch

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Mixed Precision Training Loop

```python
from mup import MuAdamW
import torch

model = create_mup_model(config=config, board_size=9, device=device)
optimizer = MuAdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
scaler = torch.amp.GradScaler()
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=500, total_steps=10000)

for batch in dataloader:
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, policy_loss, value_loss = model.compute_loss(board, move, winner)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
```

### Evaluation Function

```python
@torch.no_grad()
def evaluate(model, dataloader, device, max_samples: int = 2000) -> dict[str, float]:
    model.eval()
    total_loss, policy_correct, value_correct = 0.0, 0, 0
    n_samples, n_batches = 0, 0
    max_batches = max(1, max_samples // dataloader.batch_size)

    for batch in dataloader:
        if n_batches >= max_batches:
            break
        board = batch["board"].to(device)
        move = batch["move"].to(device)
        winner = batch["winner"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _, _ = model.compute_loss(board, move, winner)
            policy_logits, value_logits = model(board)

        total_loss += loss.item()
        target_idx = move[:, 0] * 9 + move[:, 1]
        valid_mask = move[:, 0] >= 0
        if valid_mask.sum() > 0:
            pred_idx = policy_logits[valid_mask].argmax(dim=-1)
            policy_correct += (pred_idx == target_idx[valid_mask]).sum().item()
        pred_winner = (value_logits > 0).long()
        value_correct += (pred_winner == winner).sum().item()
        n_samples += board.shape[0]
        n_batches += 1

    model.train()
    return {
        "loss": total_loss / max(1, n_batches),
        "policy_acc": policy_correct / max(1, n_samples),
        "value_acc": value_correct / max(1, n_samples),
    }
```

## Code Template

```python
"""
Experiment: <title>
Date: YYYY-MM-DD
Claude prompt: <description>
Commit: TBD
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from alpha_go.analysis import write_report_skeleton

# Constants
DATE_PREFIX = "YYYY-MM-DD_HH-MM"
SLUG = "<experiment-slug>"
EXPERIMENT_NAME = f"{DATE_PREFIX}-{SLUG}"
PROMPT = "<experiment-description>"

# Experiment directory (self-contained)
# On Ray workers: ALPHAGO_BASE_DIR=/root/alphago
BASE_DIR = Path(os.environ.get("ALPHAGO_BASE_DIR", ".")).resolve()
EXPERIMENT_DIR = BASE_DIR / "experiments" / EXPERIMENT_NAME
DATA_DIR = EXPERIMENT_DIR / "data"
FIG_DIR = EXPERIMENT_DIR / "figures"
CKPT_DIR = EXPERIMENT_DIR / "checkpoints"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Experiment implementation
    results = run_experiment()

    # Save data
    csv_path = DATA_DIR / "results.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    # Generate and save figures
    fig, ax = plt.subplots()
    # ... plotting code ...
    fig_path = FIG_DIR / "results.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Generate report skeleton for Claude to complete
    write_report_skeleton(
        slug=SLUG,
        date_prefix=DATE_PREFIX,
        experiment_dir=EXPERIMENT_DIR,
        csv_path=csv_path,
        figures=[fig_path],
        prompt=PROMPT,
    )

if __name__ == "__main__":
    main()
```
