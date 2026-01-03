---
name: hyperparameter-optimization
description: Systematic hyperparameter optimization using Optuna, Ray Tune, and Bayesian optimization techniques. Use when tuning model hyperparameters, designing search spaces, or optimizing training configurations.
---

# Hyperparameter Optimization

Master systematic hyperparameter optimization strategies for ML research, from search space design to efficient optimization algorithms.

## When to Use This Skill

- Tuning hyperparameters for new models or datasets
- Designing efficient search spaces
- Comparing optimization strategies
- Running large-scale hyperparameter sweeps
- Optimizing under compute budget constraints
- Multi-objective optimization (accuracy vs speed)

## Search Space Design

### Parameter Types and Ranges
```python
import optuna
from typing import Dict, Any

def create_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Well-designed search space with appropriate ranges."""

    config = {}

    # Learning rate: log-uniform is almost always correct
    config["learning_rate"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # Batch size: powers of 2, categorical
    config["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])

    # Weight decay: log-uniform, include 0 option
    use_weight_decay = trial.suggest_categorical("use_weight_decay", [True, False])
    if use_weight_decay:
        config["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    else:
        config["weight_decay"] = 0.0

    # Dropout: uniform between 0 and 0.5
    config["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)

    # Hidden dimensions: discrete steps
    config["hidden_dim"] = trial.suggest_int("hidden_dim", 64, 512, step=64)

    # Number of layers: small integer range
    config["n_layers"] = trial.suggest_int("n_layers", 2, 6)

    # Activation: categorical
    config["activation"] = trial.suggest_categorical(
        "activation", ["relu", "gelu", "silu", "tanh"]
    )

    # Optimizer-specific parameters
    config["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])

    if config["optimizer"] in ["adam", "adamw"]:
        config["beta1"] = trial.suggest_float("beta1", 0.85, 0.95)
        config["beta2"] = trial.suggest_float("beta2", 0.99, 0.999)
    elif config["optimizer"] == "sgd":
        config["momentum"] = trial.suggest_float("momentum", 0.8, 0.99)
        config["nesterov"] = trial.suggest_categorical("nesterov", [True, False])

    # Scheduler
    config["scheduler"] = trial.suggest_categorical(
        "scheduler", ["cosine", "linear", "constant", "warmup_cosine"]
    )

    if config["scheduler"] == "warmup_cosine":
        config["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.01, 0.1)

    return config
```

### Search Space Best Practices
```python
# DO: Use log-uniform for learning rate and weight decay
lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

# DON'T: Use linear scale for values spanning orders of magnitude
# lr = trial.suggest_float("lr", 0.00001, 0.01)  # Bad!

# DO: Use categorical for discrete choices
batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

# DON'T: Use int with step=1 for powers of 2
# batch_size = trial.suggest_int("batch_size", 32, 128)  # Tries 33, 34, etc.

# DO: Define conditional parameters
use_dropout = trial.suggest_categorical("use_dropout", [True, False])
if use_dropout:
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

# DO: Constrain dependent parameters
n_heads = trial.suggest_categorical("n_heads", [4, 8, 16])
# Hidden dim must be divisible by n_heads
hidden_dim = trial.suggest_categorical(
    "hidden_dim",
    [d for d in [256, 512, 768, 1024] if d % n_heads == 0]
)
```

## Optuna Patterns

### Basic Study Setup
```python
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler

def create_study(
    study_name: str,
    storage: str = None,
    direction: str = "maximize",
    n_startup_trials: int = 10,
) -> optuna.Study:
    """Create Optuna study with best practices."""

    # TPE sampler with startup random trials
    sampler = TPESampler(
        n_startup_trials=n_startup_trials,
        seed=42,
        multivariate=True,  # Model parameter interactions
        group=True,  # Group conditional parameters
    )

    # Hyperband pruner for early stopping
    pruner = HyperbandPruner(
        min_resource=1,       # Minimum epochs
        max_resource=100,     # Maximum epochs
        reduction_factor=3,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True,
    )

    return study


def objective(trial: optuna.Trial) -> float:
    """Objective function with pruning support."""

    config = create_search_space(trial)

    # Create model and training components
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    train_loader, val_loader = get_dataloaders(config["batch_size"])

    # Training loop with pruning
    for epoch in range(100):
        train_one_epoch(model, train_loader, optimizer)
        val_metric = evaluate(model, val_loader)

        # Report intermediate value
        trial.report(val_metric, epoch)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_metric


# Run optimization
study = create_study("my_experiment", storage="sqlite:///optuna.db")
study.optimize(objective, n_trials=100, timeout=3600)

print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### Multi-Objective Optimization
```python
def multi_objective(trial: optuna.Trial) -> tuple:
    """Optimize for accuracy AND efficiency."""

    config = create_search_space(trial)
    model = create_model(config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())

    # Train and evaluate
    train(model, config)
    accuracy = evaluate_accuracy(model)

    # Return both objectives
    return accuracy, -n_params  # Maximize accuracy, minimize params (negate)


study = optuna.create_study(
    directions=["maximize", "maximize"],  # Both maximized (params negated)
    sampler=optuna.samplers.NSGAIISampler(),
)

study.optimize(multi_objective, n_trials=100)

# Get Pareto front
print(f"Number of Pareto-optimal solutions: {len(study.best_trials)}")
for trial in study.best_trials:
    print(f"  Accuracy: {trial.values[0]:.4f}, Params: {-trial.values[1]:,}")
```

### Distributed Optimization
```python
# Worker 1, 2, 3, ... all run the same code
import optuna

storage = "mysql://user:pass@host/db"  # Shared storage

study = optuna.create_study(
    study_name="distributed_study",
    storage=storage,
    load_if_exists=True,
)

# Each worker optimizes independently, sharing results via storage
study.optimize(objective, n_trials=100)
```

## Ray Tune Integration

### Basic Ray Tune Setup
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.optuna import OptunaSearch

def ray_tune_search():
    """Distributed HPO with Ray Tune."""

    # Define search space
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "hidden_dim": tune.choice([128, 256, 512]),
        "n_layers": tune.randint(2, 6),
        "dropout": tune.uniform(0, 0.5),
    }

    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=100,        # Max epochs
        grace_period=10,  # Min epochs before pruning
        reduction_factor=3,
    )

    # Optuna search algorithm
    search_alg = OptunaSearch(
        metric="val_accuracy",
        mode="max",
    )

    # Run tuning
    analysis = tune.run(
        train_fn,  # Your training function
        config=search_space,
        num_samples=100,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={"cpu": 2, "gpu": 1},
        local_dir="./ray_results",
        name="hpo_experiment",
    )

    return analysis.best_config, analysis.best_result


def train_fn(config):
    """Training function for Ray Tune."""
    model = create_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(100):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_accuracy = evaluate(model, val_loader)

        # Report metrics to Ray Tune
        tune.report(
            train_loss=train_loss,
            val_accuracy=val_accuracy,
        )
```

### Population Based Training
```python
from ray.tune.schedulers import PopulationBasedTraining

# PBT mutates hyperparameters during training
pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="val_accuracy",
    mode="max",
    perturbation_interval=5,  # Perturb every 5 epochs
    hyperparam_mutations={
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": [32, 64, 128],
    },
)

analysis = tune.run(
    train_fn,
    scheduler=pbt,
    num_samples=8,  # Population size
    # ... other config
)
```

## Analysis and Visualization

### Optuna Visualization
```python
import optuna.visualization as vis

def analyze_study(study: optuna.Study):
    """Comprehensive study analysis."""

    # Parameter importance
    importance = optuna.importance.get_param_importances(study)
    print("\nParameter Importance:")
    for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {param}: {imp:.3f}")

    # Visualizations
    fig_history = vis.plot_optimization_history(study)
    fig_importance = vis.plot_param_importances(study)
    fig_parallel = vis.plot_parallel_coordinate(study)
    fig_slice = vis.plot_slice(study)
    fig_contour = vis.plot_contour(study, params=["lr", "weight_decay"])

    # Save figures
    fig_history.write_html("optimization_history.html")
    fig_importance.write_html("param_importance.html")
    fig_parallel.write_html("parallel_coordinate.html")

    # Get statistics
    trials_df = study.trials_dataframe()
    print(f"\nCompleted trials: {len(trials_df)}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Top 5 configurations
    print("\nTop 5 configurations:")
    top5 = trials_df.nlargest(5, 'value')[['number', 'value', 'params_lr', 'params_batch_size']]
    print(top5.to_string())

    return importance
```

### Custom Analysis
```python
def analyze_hyperparameter_sensitivity(study: optuna.Study):
    """Analyze how sensitive performance is to each hyperparameter."""

    df = study.trials_dataframe()

    # Filter completed trials
    df = df[df['state'] == 'COMPLETE']

    sensitivities = {}

    for param_col in [c for c in df.columns if c.startswith('params_')]:
        param_name = param_col.replace('params_', '')
        param_values = df[param_col]

        if param_values.dtype in ['float64', 'int64']:
            # Correlation for numeric parameters
            correlation = param_values.corr(df['value'])
            sensitivities[param_name] = abs(correlation)
        else:
            # For categorical, compute variance of means
            group_means = df.groupby(param_col)['value'].mean()
            sensitivities[param_name] = group_means.std()

    # Sort by sensitivity
    sorted_sens = sorted(sensitivities.items(), key=lambda x: -x[1])

    print("Hyperparameter Sensitivity:")
    for param, sens in sorted_sens:
        print(f"  {param}: {sens:.4f}")

    return sensitivities
```

## Budget-Aware Optimization

### Compute Budget Planning
```python
def plan_optimization(
    gpu_hours_budget: float,
    single_trial_hours: float,
    search_space_size: int,
) -> Dict[str, Any]:
    """Plan optimization strategy given budget."""

    max_trials = int(gpu_hours_budget / single_trial_hours)

    if max_trials < 20:
        strategy = "random"
        recommendation = "Limited budget: Use random search on most important params"
    elif max_trials < 50:
        strategy = "tpe"
        recommendation = "Use TPE with aggressive pruning"
    elif max_trials < 200:
        strategy = "tpe_multivariate"
        recommendation = "Use TPE with parameter interactions"
    else:
        strategy = "hyperband"
        recommendation = "Large budget: Use Hyperband with full exploration"

    return {
        "max_trials": max_trials,
        "strategy": strategy,
        "recommendation": recommendation,
        "estimated_hours": max_trials * single_trial_hours,
    }
```

### Early Stopping Integration
```python
def objective_with_early_stopping(trial: optuna.Trial) -> float:
    """Objective with patience-based early stopping."""

    config = create_search_space(trial)
    model = create_model(config)

    best_val = float('-inf')
    patience = 10
    no_improve = 0

    for epoch in range(100):
        train_one_epoch(model)
        val_metric = evaluate(model)

        # Report to Optuna
        trial.report(val_metric, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        # Patience-based early stopping
        if val_metric > best_val:
            best_val = val_metric
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_val
```

## Reproducibility

### Reproducible HPO
```python
def reproducible_optimization(seed: int = 42):
    """Fully reproducible hyperparameter optimization."""

    # Set all seeds
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create study with fixed sampler seed
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler)

    # Log configuration
    config = {
        "seed": seed,
        "sampler": "TPESampler",
        "n_trials": 100,
        "search_space": "documented in create_search_space()",
    }

    with open("hpo_config.json", "w") as f:
        json.dump(config, f)

    return study
```

## Best Practices

1. **Start with random search** for initial exploration
2. **Use log scale** for learning rate and weight decay
3. **Enable pruning** to save compute on bad trials
4. **Log everything** for reproducibility
5. **Use conditional parameters** for dependent hyperparameters
6. **Run multiple seeds** for final configuration validation

## Common Pitfalls

- **Too narrow search space**: Missing the optimal region
- **Too wide search space**: Wasting trials on bad regions
- **Ignoring interactions**: Some params depend on others
- **Overfitting to validation**: Report on held-out test set
- **Not enough trials**: Drawing conclusions from noise
