---
name: cmd-ai-ml-experiment
description: "Create and run a timestamped experiment script with self-contained folder. Use when running ML experiments, parameter sweeps, or evaluation studies. Do NOT use for data exploration (use Jupyter directly)."
disable-model-invocation: true
---

# Experiment Runner

Before creating the experiment, gather diagnostic context:

1. **Detect experiment mode** from the user's input:
   - "re-run" / "resume" → locate existing experiment folder, re-run its `experiment.py`, update report
   - "series" / "sequence" → sequential/adaptive multi-experiment workflow
   - Otherwise → single experiment (default)

2. **Check project structure** — scan for `pyproject.toml`, existing `experiments/` directory, available datasets (`game_data/`), and model checkpoints.

3. **Review recent experiments** — `ls -dt experiments/*/` (latest 5) to avoid duplicating work and to inform design.

4. **Check available infrastructure** — look for `infra/ray_cluster.py` to determine if distributed submission is available.

---

## Folder Structure

Each experiment is self-contained:
```
experiments/<datetime>-<slug>/
    experiment.py      # Main experiment script
    report.md          # Markdown report with findings
    game_data/*.npz    # GameRecord played and saved
    data/*.csv         # Result data files
    figures/*.png      # Generated figures
    checkpoints/*.pt   # Model checkpoints (if any)
```

## Single Experiment Workflow

1. **Generate timestamp and slug**: `%Y-%m-%d_%H-%M` + kebab-case slug (e.g., `2025-12-30_00-27-onestep-q-estimation`)

2. **Create experiment folder and script**:
   - Write `experiments/<datetime>-<slug>/experiment.py`
   - Save outputs to its own folder (data/, figures/, checkpoints/)
   - Report execution time

3. **Run**: `uv run experiments/<datetime>-<slug>/experiment.py`
   - For long-running experiments, save intermediate results (game_data) so partial data is usable if stopped early

4. **Analysis**: If needed, write additional analysis scripts in the same experiment folder

5. **Complete report**: The experiment script generates a skeleton via `write_report_skeleton()`. Fill in:
   - User prompt, methodology, analysis, key findings, conclusions, execution time

6. **Git commit** (script + report only, not data/figures/checkpoints):
   ```bash
   git add experiments/<datetime>-<slug>/experiment.py experiments/<datetime>-<slug>/report.md
   git commit -m "Add experiment: <experiment-slug>"
   ```

7. **Retrospective**: Note any difficulties that suggest changes to the codebase or skill files

## Sequential/Adaptive Experiments

Triggered by "series" or "sequence" requests (e.g., "run 10 experiments to tune MCTS parameters").

### Folder Structure
```
experiments/<datetime>-<research-slug>/
    README.md
    <datetime>-<exp-slug>-01/
        experiment.py, report.md, data/, figures/
    <datetime>-<exp-slug>-02/
        ...
    synthesis.md
```

### Workflow

1. **Initialize**: Create parent folder + README with research goal
2. **For each experiment**:
   - **Design** (skip for first): Review prior reports, identify unknowns, design next experiment
   - **Execute**: Create subfolder, write experiment.py with clear hypothesis, run, write report, git commit
   - **Analyze**: Extract learnings, update beliefs, identify next priority
3. **Synthesize**: Write `synthesis.md` — table of experiments, overall conclusions, recommended config, open questions

### Principles
- Test one variable at a time
- Clear hypothesis with expected outcome
- Build on prior work (reference which earlier experiments informed design)
- Adapt the plan based on findings

## Datasets

Default dataset: path expressions in `game_data/9x9/dataset.txt`

```python
# Single directory
dataset = GoDataset("game_data/9x9/dev-train")

# Multiple directories
dataset = GoDataset([
    "game_data/9x9/dataset1",
    "game_data/9x9/dataset2",
])
```

## Code Modifications

Fork `alpha_go/train.py` to the experiment file. Inline model architecture changes. Only create/edit `*.py` in `experiments/`. Ask user before editing `src/`.

## Gameplay and Evaluation

Use `alpha_go.gameplay.play_game` or `uv run alpha_go.self_play` for evaluation and data generation.

```python
from alpha_go.gameplay import play_game
record = play_game(black_agent, white_agent, log_to_db=True)
```

Query results:
```python
from alpha_go.eval_log import EvalLogDB
db = EvalLogDB.get_instance()
win_rate = db.get_win_rate_vs_baseline(
    checkpoint="checkpoints/step_50000.pt",
    baseline_agent="gnugo5",
    as_black=True,
)
```

## Distributed Execution (Ray)

### Submitting Jobs

```bash
# Basic
./infra/ray_cluster.py submit -- /root/.local/bin/uv run experiments/<slug>/experiment.py

# With GPU
./infra/ray_cluster.py submit --entrypoint-num-gpus 1 -- /root/.local/bin/uv run experiments/<slug>/experiment.py

# Non-blocking
./infra/ray_cluster.py submit --no-wait -- /root/.local/bin/uv run experiments/<slug>/experiment.py
```

### Worker Types

| Type | Resource Flag | Has | Use For |
|------|--------------|-----|---------|
| Bare metal | (default) | Python, Ray, uv | Training, pure-Python |
| Container | `--entrypoint-resources '{"container_worker": 1}'` | alpha_go_cpp, KataGo | Compiled deps, eval |

**Jobs using `alpha_go_cpp` or KataGo MUST use container workers.**

### R2 Data Dependencies

```python
from alpha_go.r2_path import R2Path

ckpt = R2Path("checkpoints/step_50.pt")            # Download on access
ckpt = R2Path("checkpoints/step_50.pt", upload=True) # Upload in background
ckpt.wait_for_upload()                                # Block until done
```

On Ray workers: `ALPHAGO_BASE_DIR=/root/alphago`

Zip `.npz` datasets before uploading to minimize API calls.

### Parallel Eval Pattern

Multi-script structure for N checkpoints x M opponents:
```
experiments/<slug>/
    01_upload_dependencies.py   # Upload to R2
    02_eval_single.py           # Single eval job (runs on Ray)
    03_submit_all.py            # Submit all jobs
    04_analyze_results.py       # Download + analyze
    data/job_manifest.json
```

1. Upload deps with `R2Path(path, upload=True)`
2. Worker script: takes CLI args, downloads via `R2Path()`, runs eval, saves to R2, prints `===RESULT===` block
3. **Test one job first** before submitting all
4. Submit all with `--no-wait`, save job IDs to `data/submitted_jobs.json`
5. Download results with `rclone sync`, generate plots

See `experiments/2026-01-25_19-24-checkpoint-pair-eval/` for complete example.

### Distributed RL (Replay Buffer + Trainer + Collector)

```bash
uv run example_recipes/distributed_collect.py
```

Driver runs locally, launches processes on remote nodes via Ray.

## Analysis Library

```python
from alpha_go.analysis import (
    load_training_run, compute_cumulative_flops,
    plot_loss_vs_flops, plot_accuracy_vs_flops, plot_training_summary,
    render_board_simple, render_board_with_policy,
    get_star_points, get_column_labels,
    write_report_skeleton,
)
```

### Report Generation
```python
write_report_skeleton(
    slug="onestep-q-estimation",
    date_prefix="2025-12-30_00-27",
    experiment_dir=EXPERIMENT_DIR,
    csv_path=DATA_DIR / "results.csv",
    figures=[FIG_DIR / "results.png"],
    prompt="Test one-step Q estimation vs raw policy",
)
```

### Board Visualization
```python
render_board_simple(board, ax, title="Game Position", board_size=9)
render_board_with_policy(
    board=board, policy_2d=policy_2d, ax=ax,
    legal_moves=legal_moves, chosen_move=(3, 3),
    title="Policy Distribution", board_size=9,
    show_probs=True, prob_threshold=0.05,
)
```

## Training Patterns

### Cosine Schedule with Warmup
```python
import math, torch

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Mixed Precision Loop
```python
from mup import MuAdamW
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
def evaluate(model, dataloader, device, max_samples=2000):
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

DATE_PREFIX = "YYYY-MM-DD_HH-MM"
SLUG = "<experiment-slug>"
EXPERIMENT_NAME = f"{DATE_PREFIX}-{SLUG}"
PROMPT = "<experiment-description>"

BASE_DIR = Path(os.environ.get("ALPHAGO_BASE_DIR", ".")).resolve()
EXPERIMENT_DIR = BASE_DIR / "experiments" / EXPERIMENT_NAME
DATA_DIR = EXPERIMENT_DIR / "data"
FIG_DIR = EXPERIMENT_DIR / "figures"
CKPT_DIR = EXPERIMENT_DIR / "checkpoints"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    results = run_experiment()
    csv_path = DATA_DIR / "results.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots()
    # ... plotting ...
    fig_path = FIG_DIR / "results.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    write_report_skeleton(
        slug=SLUG, date_prefix=DATE_PREFIX,
        experiment_dir=EXPERIMENT_DIR,
        csv_path=csv_path, figures=[fig_path],
        prompt=PROMPT,
    )

if __name__ == "__main__":
    main()
```

### Expected CSV Schema

Training CSVs should include: `global_step`, `train_loss`, `val_loss`, `train_eval_loss`, `train_policy_acc`, `val_policy_acc`, `train_value_acc`, `val_value_acc`, `batch_size` (optional, defaults to 64).

Companion flops file: `<run_name>_flops.json` → `{"n_params": 1000000, "board_size": 9}`

Now create and run the experiment described by the user.
