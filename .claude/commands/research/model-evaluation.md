# Model Evaluation and Benchmarking

Comprehensive model evaluation with proper metrics, statistical analysis, and comparison against baselines.

## Requirements

- **Evaluation context**: $ARGUMENTS (provide model type, task, datasets, and baselines to compare against)

## Instructions

### 1. Evaluation Framework Setup

**A. Metrics Selection by Task**

| Task | Primary Metrics | Secondary Metrics |
|------|-----------------|-------------------|
| Classification | Accuracy, F1, AUC-ROC | Precision, Recall, MCC |
| Object Detection | mAP@0.5, mAP@0.5:0.95 | AR, per-class AP |
| Segmentation | mIoU, Dice | Boundary F1, pixel accuracy |
| NLP Generation | BLEU, ROUGE, BERTScore | Perplexity, METEOR |
| Regression | MSE, MAE, R² | MAPE, Huber loss |
| Ranking | NDCG, MRR, MAP | Precision@K, Recall@K |
| Clustering | Silhouette, ARI, NMI | Purity, V-measure |

**B. Evaluation Code Template:**
```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn import metrics
import torch

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metrics: Dict[str, float]
    predictions: np.ndarray
    targets: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"EvaluationResult({metric_str})"


class ModelEvaluator:
    """Comprehensive model evaluation framework."""

    def __init__(self, task: str = "classification"):
        self.task = task
        self.metric_functions = self._get_metric_functions()

    def _get_metric_functions(self) -> Dict:
        """Get appropriate metrics for the task."""
        if self.task == "classification":
            return {
                "accuracy": lambda y, p: metrics.accuracy_score(y, p.argmax(-1)),
                "f1_macro": lambda y, p: metrics.f1_score(y, p.argmax(-1), average='macro'),
                "f1_weighted": lambda y, p: metrics.f1_score(y, p.argmax(-1), average='weighted'),
            }
        elif self.task == "binary_classification":
            return {
                "accuracy": lambda y, p: metrics.accuracy_score(y, (p > 0.5).astype(int)),
                "auc_roc": lambda y, p: metrics.roc_auc_score(y, p),
                "f1": lambda y, p: metrics.f1_score(y, (p > 0.5).astype(int)),
                "precision": lambda y, p: metrics.precision_score(y, (p > 0.5).astype(int)),
                "recall": lambda y, p: metrics.recall_score(y, (p > 0.5).astype(int)),
            }
        elif self.task == "regression":
            return {
                "mse": lambda y, p: metrics.mean_squared_error(y, p),
                "mae": lambda y, p: metrics.mean_absolute_error(y, p),
                "r2": lambda y, p: metrics.r2_score(y, p),
            }
        else:
            raise ValueError(f"Unknown task: {self.task}")

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda"
    ) -> EvaluationResult:
        """Run evaluation on a dataset."""
        model.eval()
        all_predictions = []
        all_targets = []

        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)

            outputs = model(inputs)
            predictions = outputs.cpu().numpy()

            all_predictions.append(predictions)
            all_targets.append(targets.numpy())

        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        # Compute all metrics
        computed_metrics = {}
        for name, func in self.metric_functions.items():
            try:
                computed_metrics[name] = float(func(targets, predictions))
            except Exception as e:
                computed_metrics[name] = float('nan')
                print(f"Warning: Could not compute {name}: {e}")

        return EvaluationResult(
            metrics=computed_metrics,
            predictions=predictions,
            targets=targets,
        )
```

### 2. Statistical Significance Testing

```python
import scipy.stats as stats
from typing import Tuple

def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Paired bootstrap test for comparing two models.

    Returns:
        p_value: Probability that B is not better than A
        mean_diff: Mean difference (B - A)
        ci: Confidence interval for the difference
    """
    n = len(scores_a)
    assert len(scores_b) == n

    observed_diff = scores_b.mean() - scores_a.mean()

    # Bootstrap
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        diff = scores_b[idx].mean() - scores_a[idx].mean()
        diffs.append(diff)

    diffs = np.array(diffs)

    # P-value (two-tailed)
    p_value = (np.abs(diffs) >= np.abs(observed_diff)).mean()

    # Confidence interval
    alpha = 1 - confidence
    ci = (np.percentile(diffs, 100 * alpha / 2),
          np.percentile(diffs, 100 * (1 - alpha / 2)))

    return p_value, observed_diff, ci


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """
    McNemar's test for comparing classifier accuracy.

    Args:
        correct_a: Boolean array, True where model A is correct
        correct_b: Boolean array, True where model B is correct

    Returns:
        p_value from McNemar's test
    """
    # Count disagreements
    b_correct_a_wrong = ((~correct_a) & correct_b).sum()
    a_correct_b_wrong = (correct_a & (~correct_b)).sum()

    # McNemar's test with continuity correction
    statistic = (abs(b_correct_a_wrong - a_correct_b_wrong) - 1) ** 2
    statistic /= (b_correct_a_wrong + a_correct_b_wrong)

    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return p_value


def multi_seed_significance(
    results_a: List[float],
    results_b: List[float]
) -> Dict[str, float]:
    """
    Compare models run with multiple seeds.

    Returns dict with t-test and Wilcoxon test results.
    """
    results = {}

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(results_a, results_b)
    results['t_test_pvalue'] = t_pval
    results['t_statistic'] = t_stat

    # Wilcoxon signed-rank test (non-parametric)
    if len(results_a) >= 6:  # Minimum for Wilcoxon
        w_stat, w_pval = stats.wilcoxon(results_a, results_b)
        results['wilcoxon_pvalue'] = w_pval

    # Effect size (Cohen's d)
    diff = np.array(results_b) - np.array(results_a)
    cohens_d = diff.mean() / diff.std()
    results['cohens_d'] = cohens_d

    return results
```

### 3. Multi-Dataset Evaluation

```python
from pathlib import Path
import json
from datetime import datetime

class BenchmarkSuite:
    """Run evaluation across multiple datasets."""

    def __init__(self, datasets: Dict[str, torch.utils.data.DataLoader]):
        self.datasets = datasets
        self.results = {}

    def run_benchmark(
        self,
        model: torch.nn.Module,
        model_name: str,
        evaluator: ModelEvaluator,
        device: str = "cuda"
    ) -> Dict[str, EvaluationResult]:
        """Evaluate model on all datasets."""
        results = {}

        for dataset_name, dataloader in self.datasets.items():
            print(f"Evaluating on {dataset_name}...")
            result = evaluator.evaluate(model, dataloader, device)
            results[dataset_name] = result

            # Print summary
            for metric, value in result.metrics.items():
                print(f"  {metric}: {value:.4f}")

        self.results[model_name] = results
        return results

    def compare_models(self) -> str:
        """Generate comparison table."""
        if not self.results:
            return "No results to compare"

        # Build comparison table
        lines = ["| Model | Dataset | " + " | ".join(
            list(list(self.results.values())[0].values())[0].metrics.keys()
        ) + " |"]
        lines.append("|" + "---|" * (len(lines[0].split("|")) - 2))

        for model_name, dataset_results in self.results.items():
            for dataset_name, result in dataset_results.items():
                values = " | ".join(f"{v:.4f}" for v in result.metrics.values())
                lines.append(f"| {model_name} | {dataset_name} | {values} |")

        return "\n".join(lines)

    def save_results(self, path: str):
        """Save results to JSON."""
        serializable = {}
        for model, datasets in self.results.items():
            serializable[model] = {
                ds: {"metrics": res.metrics}
                for ds, res in datasets.items()
            }

        with open(path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": serializable
            }, f, indent=2)
```

### 4. Error Analysis

```python
class ErrorAnalyzer:
    """Analyze model errors and failure modes."""

    def __init__(self, predictions: np.ndarray, targets: np.ndarray):
        self.predictions = predictions
        self.targets = targets
        self.pred_labels = predictions.argmax(-1) if predictions.ndim > 1 else predictions

    def confusion_matrix(self, class_names: Optional[List[str]] = None):
        """Generate confusion matrix with analysis."""
        cm = metrics.confusion_matrix(self.targets, self.pred_labels)

        # Find most confused pairs
        np.fill_diagonal(cm, 0)
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if cm[i, j] > 0:
                    confused_pairs.append((i, j, cm[i, j]))

        confused_pairs.sort(key=lambda x: -x[2])

        return {
            "matrix": cm,
            "most_confused": confused_pairs[:10],
            "class_names": class_names
        }

    def error_by_confidence(self, n_bins: int = 10):
        """Analyze errors by prediction confidence."""
        confidences = self.predictions.max(-1) if self.predictions.ndim > 1 else self.predictions
        correct = (self.pred_labels == self.targets)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_accuracies = []
        bin_counts = []

        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if mask.sum() > 0:
                bin_accuracies.append(correct[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)

        return {
            "bins": bins,
            "accuracies": bin_accuracies,
            "counts": bin_counts,
            "ece": self._expected_calibration_error(confidences, correct, bins)
        }

    def _expected_calibration_error(self, confidences, correct, bins):
        """Compute Expected Calibration Error."""
        ece = 0
        n = len(confidences)
        for i in range(len(bins) - 1):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if mask.sum() > 0:
                acc = correct[mask].mean()
                conf = confidences[mask].mean()
                ece += mask.sum() / n * abs(acc - conf)
        return ece

    def find_hard_examples(self, n: int = 100) -> np.ndarray:
        """Find examples the model struggles with most."""
        if self.predictions.ndim > 1:
            # Multi-class: low confidence on correct class
            correct_class_probs = self.predictions[
                np.arange(len(self.targets)), self.targets
            ]
            hardest_idx = np.argsort(correct_class_probs)[:n]
        else:
            # Binary/regression: largest errors
            errors = np.abs(self.predictions - self.targets)
            hardest_idx = np.argsort(-errors)[:n]

        return hardest_idx
```

### 5. Reporting Template

```python
def generate_evaluation_report(
    model_name: str,
    results: Dict[str, EvaluationResult],
    baseline_results: Optional[Dict[str, EvaluationResult]] = None
) -> str:
    """Generate markdown evaluation report."""

    report = f"# Evaluation Report: {model_name}\n\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

    report += "## Summary\n\n"
    report += "| Dataset | " + " | ".join(list(results.values())[0].metrics.keys()) + " |\n"
    report += "|" + "---|" * (len(list(results.values())[0].metrics) + 1) + "\n"

    for dataset, result in results.items():
        values = " | ".join(f"{v:.4f}" for v in result.metrics.values())
        report += f"| {dataset} | {values} |\n"

    if baseline_results:
        report += "\n## Comparison with Baseline\n\n"
        for dataset in results:
            if dataset in baseline_results:
                report += f"\n### {dataset}\n"
                for metric in results[dataset].metrics:
                    ours = results[dataset].metrics[metric]
                    base = baseline_results[dataset].metrics[metric]
                    diff = ours - base
                    symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                    report += f"- {metric}: {ours:.4f} ({symbol} {abs(diff):.4f})\n"

    return report
```

## Output Format

Provide:
1. **Metrics selection** appropriate for the task
2. **Evaluation code** tailored to model/task
3. **Statistical tests** for comparing against baselines
4. **Results table** with all metrics
5. **Error analysis** identifying failure modes
6. **Visualization suggestions** for results presentation
7. **Recommendations** for improvement based on analysis
