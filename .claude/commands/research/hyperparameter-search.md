# Hyperparameter Optimization

Systematic hyperparameter search with modern optimization strategies, proper validation, and reproducible results.

## Requirements

- **Optimization context**: $ARGUMENTS (provide model type, search space description, compute budget, and optimization goals)

## Instructions

### 1. Search Strategy Selection

**A. Strategy Comparison:**

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| Grid Search | Small spaces (<100 configs) | Exhaustive, reproducible | Exponential scaling |
| Random Search | Medium spaces, limited budget | Better coverage per trial | May miss optima |
| Bayesian (TPE/GP) | Expensive evaluations | Sample efficient | Overhead, local optima |
| Hyperband/ASHA | Neural networks | Early stopping, efficient | Needs epoch-wise metrics |
| Population-Based | RL, large models | Adaptive schedules | Complex, high compute |

**B. Framework Selection:**
- **Optuna**: Best overall, flexible, good visualization
- **Ray Tune**: Distributed, integrates with many frameworks
- **Weights & Biases Sweeps**: Integrated with tracking
- **Keras Tuner**: Best for Keras/TensorFlow
- **scikit-optimize**: Simple Bayesian optimization

### 2. Search Space Definition

```python
import optuna
from typing import Dict, Any, Callable
from dataclasses import dataclass
import numpy as np

@dataclass
class SearchSpace:
    """Define hyperparameter search space."""

    @staticmethod
    def suggest_learning_rate(trial: optuna.Trial) -> float:
        """Log-uniform learning rate."""
        return trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    @staticmethod
    def suggest_batch_size(trial: optuna.Trial) -> int:
        """Power-of-2 batch sizes."""
        return trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])

    @staticmethod
    def suggest_optimizer(trial: optuna.Trial) -> Dict[str, Any]:
        """Optimizer with conditional parameters."""
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])

        config = {"name": optimizer_name}

        if optimizer_name in ["adam", "adamw"]:
            config["betas"] = (
                trial.suggest_float("beta1", 0.8, 0.99),
                trial.suggest_float("beta2", 0.9, 0.999)
            )

        if optimizer_name == "adamw":
            config["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

        if optimizer_name == "sgd":
            config["momentum"] = trial.suggest_float("momentum", 0.8, 0.99)
            config["nesterov"] = trial.suggest_categorical("nesterov", [True, False])

        return config

    @staticmethod
    def suggest_architecture(trial: optuna.Trial) -> Dict[str, Any]:
        """Neural network architecture parameters."""
        return {
            "n_layers": trial.suggest_int("n_layers", 2, 6),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
        }


# Complete search space example
def create_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Full hyperparameter configuration."""

    config = {
        # Training
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs": 100,  # Fixed, use early stopping

        # Optimizer
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),

        # Architecture
        "hidden_dim": trial.suggest_int("hidden_dim", 64, 512, step=64),
        "n_layers": trial.suggest_int("n_layers", 2, 5),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),

        # Regularization
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),

        # Scheduler
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "linear", "constant"]),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.1),
    }

    return config
```

### 3. Optuna Study Setup

```python
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler
import torch
from typing import Optional
import logging

def create_study(
    study_name: str,
    direction: str = "maximize",
    storage: Optional[str] = None,
    n_startup_trials: int = 10,
    pruner_type: str = "hyperband"
) -> optuna.Study:
    """Create Optuna study with best practices."""

    # Sampler: TPE with startup random trials
    sampler = TPESampler(
        n_startup_trials=n_startup_trials,
        seed=42,
        multivariate=True,  # Consider parameter interactions
    )

    # Pruner for early stopping
    if pruner_type == "median":
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    else:  # hyperband
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=100,
            reduction_factor=3
        )

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    return study


def objective(trial: optuna.Trial) -> float:
    """Objective function with pruning support."""

    # Get hyperparameters
    config = create_search_space(trial)

    # Create model and train
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    train_loader, val_loader = get_dataloaders(config["batch_size"])

    for epoch in range(config["epochs"]):
        train_one_epoch(model, train_loader, optimizer)
        val_metric = evaluate(model, val_loader)

        # Report intermediate value for pruning
        trial.report(val_metric, epoch)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_metric


def run_optimization(
    n_trials: int = 100,
    timeout: Optional[int] = None,
    n_jobs: int = 1
):
    """Run hyperparameter optimization."""

    study = create_study(
        study_name="my_experiment",
        direction="maximize",
        storage="sqlite:///optuna.db"  # Persist results
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # Print results
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study
```

### 4. Multi-Objective Optimization

```python
def multi_objective(trial: optuna.Trial) -> tuple:
    """Optimize for both accuracy and efficiency."""

    config = create_search_space(trial)
    model = create_model(config)

    # Train and evaluate
    train(model, config)
    accuracy = evaluate_accuracy(model)
    latency = measure_latency(model)

    # Return multiple objectives
    return accuracy, -latency  # Maximize accuracy, minimize latency


def run_multi_objective():
    """Run multi-objective optimization."""

    study = optuna.create_study(
        directions=["maximize", "maximize"],  # accuracy up, latency down (negated)
        sampler=optuna.samplers.NSGAIISampler()
    )

    study.optimize(multi_objective, n_trials=100)

    # Get Pareto front
    pareto_trials = study.best_trials
    print(f"Found {len(pareto_trials)} Pareto-optimal configurations")

    return study
```

### 5. Distributed Optimization

```python
# With Ray Tune
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

def ray_tune_search():
    """Distributed hyperparameter search with Ray Tune."""

    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "hidden_dim": tune.choice([128, 256, 512]),
        "dropout": tune.uniform(0, 0.4),
    }

    scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=3
    )

    search_alg = OptunaSearch(metric="val_accuracy", mode="max")

    analysis = tune.run(
        train_fn,
        config=search_space,
        num_samples=100,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={"cpu": 2, "gpu": 1},
        local_dir="./ray_results",
    )

    return analysis.best_config
```

### 6. Analysis and Visualization

```python
def analyze_study(study: optuna.Study):
    """Comprehensive study analysis."""

    # Parameter importance
    importance = optuna.importance.get_param_importances(study)
    print("\nParameter Importance:")
    for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {param}: {imp:.3f}")

    # Visualizations
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
        plot_contour
    )

    # Create plots
    fig_history = plot_optimization_history(study)
    fig_importance = plot_param_importances(study)
    fig_parallel = plot_parallel_coordinate(study)

    # Get best trials analysis
    trials_df = study.trials_dataframe()

    # Find configurations that consistently perform well
    top_trials = trials_df.nlargest(10, 'value')
    print("\nTop 10 configurations:")
    print(top_trials[['number', 'value'] + [c for c in top_trials.columns if c.startswith('params_')]])

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "importance": importance,
        "top_trials": top_trials,
    }
```

### 7. Reproducibility Best Practices

```python
def reproducible_search(seed: int = 42):
    """Fully reproducible hyperparameter search."""

    import random
    import numpy as np
    import torch

    # Set all seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create study with fixed seed
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler)

    # Log configuration
    config_log = {
        "seed": seed,
        "sampler": "TPESampler",
        "pruner": "MedianPruner",
        "n_trials": 100,
        "search_space": {
            "learning_rate": "loguniform(1e-5, 1e-2)",
            "batch_size": "categorical([32, 64, 128])",
            # ... document full search space
        }
    }

    with open("hpo_config.json", "w") as f:
        json.dump(config_log, f, indent=2)

    return study
```

### 8. Budget-Aware Strategies

```python
def budget_aware_search(total_gpu_hours: float, single_trial_hours: float):
    """Plan search given compute budget."""

    max_trials = int(total_gpu_hours / single_trial_hours)

    print(f"Budget: {total_gpu_hours} GPU-hours")
    print(f"Estimated trials: {max_trials}")

    # Recommendations based on budget
    if max_trials < 20:
        strategy = "random_search"
        recommendation = "Limited budget: Use random search, focus on most important hyperparameters"
    elif max_trials < 100:
        strategy = "bayesian"
        recommendation = "Medium budget: Use Bayesian optimization (TPE)"
    else:
        strategy = "hyperband"
        recommendation = "Large budget: Use Hyperband/ASHA with early stopping"

    print(f"Recommended strategy: {strategy}")
    print(f"Recommendation: {recommendation}")

    return {
        "max_trials": max_trials,
        "strategy": strategy,
        "recommendation": recommendation
    }
```

## Output Format

Provide:
1. **Search space definition** - Parameters, ranges, and types
2. **Strategy recommendation** - Based on budget and problem
3. **Complete optimization code** - Ready to run
4. **Pruning/early stopping** - For efficient search
5. **Analysis code** - To interpret results
6. **Best configuration** - With confidence intervals
7. **Reproducibility setup** - Seeds and logging
8. **Visualization plan** - Key plots to generate
