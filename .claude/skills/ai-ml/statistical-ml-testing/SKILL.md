---
name: statistical-ml-testing
description: Apply proper statistical testing for ML experiments including significance tests, confidence intervals, and multiple comparison corrections. Use when comparing models, validating improvements, or publishing research results.
---

# Statistical Testing for ML

Master statistical methods for rigorous ML experiment evaluation, from significance testing to effect size estimation.

## When to Use This Skill

- Comparing model performance across configurations
- Determining if improvements are statistically significant
- Reporting results with proper uncertainty estimates
- Running multiple comparisons (ablations, hyperparameter sweeps)
- Validating that results generalize beyond test set variance
- Publishing research with rigorous statistical claims

## Basic Statistical Tests

### Paired t-test (Same Test Set)
```python
import numpy as np
from scipy import stats
from typing import Tuple, Dict

def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Paired t-test for comparing two models on the same test samples.

    Use when: Same test examples evaluated by both models.
    Assumes: Normally distributed differences.

    Args:
        scores_a: Per-sample scores from model A
        scores_b: Per-sample scores from model B
        alpha: Significance level

    Returns:
        Dictionary with test statistics and interpretation
    """
    assert len(scores_a) == len(scores_b), "Must have same number of samples"

    # Compute differences
    differences = scores_b - scores_a

    # Run paired t-test
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

    # Effect size (Cohen's d for paired samples)
    d = differences.mean() / differences.std()

    # Confidence interval for mean difference
    se = differences.std() / np.sqrt(len(differences))
    ci = stats.t.interval(1 - alpha, len(differences) - 1,
                          loc=differences.mean(), scale=se)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "is_significant": p_value < alpha,
        "mean_difference": differences.mean(),
        "std_difference": differences.std(),
        "cohens_d": d,
        "effect_size": interpret_cohens_d(d),
        "confidence_interval": ci,
        "model_b_better": differences.mean() > 0,
    }


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"
```

### Independent t-test (Different Test Sets)
```python
def independent_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Independent t-test for comparing models on different samples.

    Use when: Different random seeds, different data splits.
    Assumes: Normal distributions, equal variances (can relax).
    """
    # Welch's t-test (doesn't assume equal variances)
    t_stat, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)

    # Effect size
    pooled_std = np.sqrt(
        ((len(scores_a) - 1) * scores_a.std()**2 +
         (len(scores_b) - 1) * scores_b.std()**2) /
        (len(scores_a) + len(scores_b) - 2)
    )
    d = (scores_b.mean() - scores_a.mean()) / pooled_std

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "is_significant": p_value < alpha,
        "mean_a": scores_a.mean(),
        "mean_b": scores_b.mean(),
        "difference": scores_b.mean() - scores_a.mean(),
        "cohens_d": d,
        "effect_size": interpret_cohens_d(d),
    }
```

### McNemar's Test (Classification Agreement)
```python
def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    McNemar's test for comparing classifier accuracy.

    Use when: Comparing two classifiers on same test set.
    Tests whether they make different types of errors.

    Args:
        correct_a: Boolean array, True where model A is correct
        correct_b: Boolean array, True where model B is correct
    """
    # Build contingency table
    #                Model B correct  Model B wrong
    # Model A correct      n11            n10
    # Model A wrong        n01            n00

    n01 = ((~correct_a) & correct_b).sum()  # A wrong, B correct
    n10 = (correct_a & (~correct_b)).sum()  # A correct, B wrong

    # McNemar's test (with continuity correction)
    if n01 + n10 == 0:
        return {"p_value": 1.0, "is_significant": False, "message": "No disagreements"}

    chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "chi2_statistic": chi2,
        "p_value": p_value,
        "is_significant": p_value < alpha,
        "n_a_correct_b_wrong": int(n10),
        "n_a_wrong_b_correct": int(n01),
        "model_b_better": n01 > n10,
    }
```

## Bootstrap Methods

### Paired Bootstrap Test
```python
def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap test for comparing two models.

    More robust than t-test, doesn't assume normality.

    Args:
        scores_a: Per-sample scores from model A
        scores_b: Per-sample scores from model B
        n_bootstrap: Number of bootstrap iterations
        alpha: Significance level
    """
    np.random.seed(seed)
    n = len(scores_a)

    # Observed difference
    observed_diff = scores_b.mean() - scores_a.mean()

    # Bootstrap
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.randint(0, n, size=n)
        boot_diff = scores_b[idx].mean() - scores_a[idx].mean()
        bootstrap_diffs.append(boot_diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # P-value (two-tailed): proportion of bootstrap diffs >= observed
    # Under null hypothesis (no difference), we center at 0
    centered_diffs = bootstrap_diffs - bootstrap_diffs.mean()
    p_value = (np.abs(centered_diffs) >= np.abs(observed_diff)).mean()

    # Confidence interval (percentile method)
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return {
        "observed_difference": observed_diff,
        "p_value": p_value,
        "is_significant": p_value < alpha,
        "confidence_interval": (ci_lower, ci_upper),
        "bootstrap_mean": bootstrap_diffs.mean(),
        "bootstrap_std": bootstrap_diffs.std(),
        "model_b_better": observed_diff > 0,
    }


def bootstrap_confidence_interval(
    scores: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for any statistic.

    Args:
        scores: Sample data
        statistic: Function to compute (e.g., np.mean, np.median)
        n_bootstrap: Number of bootstrap iterations
        alpha: Significance level (0.05 for 95% CI)
    """
    np.random.seed(seed)
    n = len(scores)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        bootstrap_stats.append(statistic(scores[idx]))

    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper
```

## Multiple Comparison Corrections

### Bonferroni Correction
```python
def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Bonferroni correction for multiple comparisons.

    Most conservative: controls family-wise error rate (FWER).
    Use when: Each comparison is important, false positives costly.
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    adjusted_p = np.minimum(p_values * n_tests, 1.0)

    return {
        "adjusted_p_values": adjusted_p,
        "adjusted_alpha": adjusted_alpha,
        "significant": adjusted_p < alpha,
        "n_significant": (adjusted_p < alpha).sum(),
    }
```

### Holm-Bonferroni (Step-Down)
```python
def holm_bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Holm-Bonferroni step-down procedure.

    Less conservative than Bonferroni, still controls FWER.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Adjusted p-values
    adjusted_p = np.zeros(n)
    for i, p in enumerate(sorted_p):
        adjusted_p[sorted_idx[i]] = p * (n - i)

    # Ensure monotonicity
    adjusted_p = np.maximum.accumulate(adjusted_p[sorted_idx])[np.argsort(sorted_idx)]
    adjusted_p = np.minimum(adjusted_p, 1.0)

    return {
        "adjusted_p_values": adjusted_p,
        "significant": adjusted_p < alpha,
        "n_significant": (adjusted_p < alpha).sum(),
    }
```

### Benjamini-Hochberg (FDR Control)
```python
def benjamini_hochberg_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Benjamini-Hochberg procedure for FDR control.

    Controls False Discovery Rate instead of FWER.
    Use when: Exploratory analysis, can tolerate some false positives.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH adjusted p-values
    adjusted_p = np.zeros(n)
    for i, p in enumerate(sorted_p):
        adjusted_p[sorted_idx[i]] = p * n / (i + 1)

    # Ensure monotonicity (cumulative minimum from the end)
    adjusted_p_sorted = adjusted_p[sorted_idx]
    for i in range(n - 2, -1, -1):
        adjusted_p_sorted[i] = min(adjusted_p_sorted[i], adjusted_p_sorted[i + 1])
    adjusted_p = adjusted_p_sorted[np.argsort(sorted_idx)]
    adjusted_p = np.minimum(adjusted_p, 1.0)

    return {
        "adjusted_p_values": adjusted_p,
        "significant": adjusted_p < alpha,
        "n_significant": (adjusted_p < alpha).sum(),
        "estimated_fdr": alpha,
    }
```

## Multi-Seed Experiments

### Aggregating Results Across Seeds
```python
def multi_seed_comparison(
    results_a: list,  # List of scores from model A across seeds
    results_b: list,  # List of scores from model B across seeds
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Compare models trained with multiple random seeds.

    Args:
        results_a: List of final metrics from model A (one per seed)
        results_b: List of final metrics from model B (one per seed)
    """
    results_a = np.array(results_a)
    results_b = np.array(results_b)

    results = {
        "mean_a": results_a.mean(),
        "std_a": results_a.std(),
        "mean_b": results_b.mean(),
        "std_b": results_b.std(),
        "n_seeds": len(results_a),
    }

    # Paired test if same seeds used
    if len(results_a) == len(results_b):
        t_stat, p_value = stats.ttest_rel(results_a, results_b)
        results["paired_t_pvalue"] = p_value

    # Independent test
    t_stat, p_value = stats.ttest_ind(results_a, results_b, equal_var=False)
    results["independent_t_pvalue"] = p_value

    # Non-parametric: Wilcoxon signed-rank (paired) or Mann-Whitney U
    if len(results_a) >= 6:  # Minimum for meaningful test
        try:
            w_stat, w_pvalue = stats.wilcoxon(results_a, results_b)
            results["wilcoxon_pvalue"] = w_pvalue
        except ValueError:
            pass  # All differences may be zero

        u_stat, u_pvalue = stats.mannwhitneyu(results_a, results_b, alternative='two-sided')
        results["mann_whitney_pvalue"] = u_pvalue

    # Effect size
    pooled_std = np.sqrt((results_a.std()**2 + results_b.std()**2) / 2)
    if pooled_std > 0:
        results["cohens_d"] = (results_b.mean() - results_a.mean()) / pooled_std
        results["effect_size"] = interpret_cohens_d(results["cohens_d"])

    return results


def required_seeds_for_power(
    expected_effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """
    Calculate number of seeds needed to detect an effect.

    Args:
        expected_effect_size: Expected Cohen's d
        power: Desired statistical power (typically 0.8)
        alpha: Significance level
    """
    from scipy.stats import norm

    # Using approximation for paired t-test
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    n = ((z_alpha + z_beta) / expected_effect_size) ** 2

    return int(np.ceil(n))
```

## Reporting Results

### Result Formatting
```python
def format_result_with_uncertainty(
    mean: float,
    std: float,
    n: int,
    confidence: float = 0.95,
    decimals: int = 2,
) -> str:
    """
    Format result with confidence interval.

    Returns: "mean ± margin (95% CI: [lower, upper])"
    """
    # Standard error
    se = std / np.sqrt(n)

    # t-value for confidence interval
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_val * se

    ci_lower = mean - margin
    ci_upper = mean + margin

    return (
        f"{mean:.{decimals}f} ± {margin:.{decimals}f} "
        f"({int(confidence*100)}% CI: [{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}])"
    )


def create_results_table(
    results: Dict[str, Dict[str, float]],
    metric_name: str = "Accuracy",
) -> str:
    """Create markdown results table with statistics."""

    lines = [
        f"| Model | {metric_name} | 95% CI | Seeds |",
        "|-------|------------|--------|-------|",
    ]

    for model_name, scores in results.items():
        mean = scores["mean"]
        std = scores["std"]
        n = scores["n_seeds"]

        ci_lower, ci_upper = bootstrap_confidence_interval(
            np.array(scores["raw_scores"]),
            n_bootstrap=10000,
        )

        lines.append(
            f"| {model_name} | {mean:.2f} ± {std:.2f} | "
            f"[{ci_lower:.2f}, {ci_upper:.2f}] | {n} |"
        )

    return "\n".join(lines)
```

### Significance Matrix
```python
def create_significance_matrix(
    model_scores: Dict[str, np.ndarray],
    test_fn: callable = paired_t_test,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, list]:
    """
    Create pairwise significance matrix for multiple models.

    Returns matrix where entry [i,j] indicates if model i is
    significantly better than model j.
    """
    models = list(model_scores.keys())
    n = len(models)
    p_matrix = np.ones((n, n))
    sig_matrix = np.zeros((n, n), dtype=bool)

    # Collect all p-values for correction
    all_p_values = []
    p_indices = []

    for i in range(n):
        for j in range(i + 1, n):
            result = test_fn(model_scores[models[i]], model_scores[models[j]])
            p_matrix[i, j] = result["p_value"]
            p_matrix[j, i] = result["p_value"]
            all_p_values.append(result["p_value"])
            p_indices.append((i, j))

    # Apply multiple comparison correction
    if all_p_values:
        corrected = benjamini_hochberg_correction(np.array(all_p_values), alpha)

        for idx, (i, j) in enumerate(p_indices):
            if corrected["significant"][idx]:
                # Determine which is better
                if model_scores[models[i]].mean() > model_scores[models[j]].mean():
                    sig_matrix[i, j] = True
                else:
                    sig_matrix[j, i] = True

    return sig_matrix, models, p_matrix
```

## Best Practices

1. **Report effect sizes**, not just p-values
2. **Use confidence intervals** for uncertainty quantification
3. **Correct for multiple comparisons** in ablations
4. **Run sufficient seeds** (5-10 minimum for significance)
5. **Use appropriate tests**: paired when applicable, non-parametric when assumptions violated
6. **Pre-register hypotheses** when possible to avoid p-hacking

## Common Pitfalls

- **P-hacking**: Running tests until finding significance
- **Ignoring multiple comparisons**: Inflated false positive rate
- **Small sample sizes**: Underpowered experiments
- **Misinterpreting p-values**: P-value is NOT P(hypothesis is true)
- **Reporting only significant results**: Publication bias
- **Assuming normality**: Use non-parametric tests when violated
