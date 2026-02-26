# Causal Inference & Probabilistic Programming

## Causal Method Selection

| Scenario | Key Assumption | Method | Library |
|----------|----------------|--------|---------|
| RCT / A/B test | Randomization | Difference-in-means | scipy |
| Observational, known confounders | Unconfoundedness | T/S/X-Learner, CausalForestDML | EconML |
| Observational, high-dim confounders | Unconfoundedness + sparsity | DML (Double ML) | EconML |
| Selection on observables | Strong ignorability | IPW / Hajek estimator | DoWhy |
| Unobserved confounders, instrument exists | Valid instrument | IV / 2SLS / DMLIV | EconML |
| Pre/post with control group | Parallel trends | Difference-in-Differences | statsmodels |
| Full causal graph validation | Graph structure known | DoWhy (identify, estimate, refute) | DoWhy |
| Targeting / personalization | Unconfoundedness | Uplift modeling + Qini curves | EconML |

## DoWhy Pipeline

```python
from dowhy import CausalModel

def run_dowhy_pipeline(df, treatment, outcome, confounders,
                       instruments=None, effect_modifiers=None):
    """Full DoWhy pipeline: model, identify, estimate, refute."""
    model = CausalModel(
        data=df, treatment=treatment, outcome=outcome,
        common_causes=confounders, instruments=instruments,
        effect_modifiers=effect_modifiers,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.econml.metalearner_tlearner",
        method_params={"init_params": {"models": "GradientBoostingRegressor()"}},
    )
    # Refutation tests (critical -- skip at your peril)
    refutations = {}
    for name, method in [
        ("random_cause", "random_common_cause"),
        ("placebo", "placebo_treatment_refuter"),
        ("subset", "data_subset_refuter"),
    ]:
        kwargs = {"placebo_type": "permute"} if name == "placebo" else {}
        if name == "subset":
            kwargs = {"subset_fraction": 0.8, "num_simulations": 5}
        refutations[name] = model.refute_estimate(
            estimand, estimate, method_name=method, **kwargs,
        )
    return estimate, refutations
```

## EconML Meta-Learners

```python
from econml.metalearners import TLearner, SLearner, XLearner
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

def compare_metalearners(X, T, Y):
    """Compare T/S/X-Learner for CATE estimation."""
    models = {
        "T": TLearner(models=GradientBoostingRegressor(n_estimators=200)),
        "S": SLearner(overall_model=GradientBoostingRegressor(n_estimators=200)),
        "X": XLearner(
            models=GradientBoostingRegressor(n_estimators=200),
            propensity_model=GradientBoostingRegressor(n_estimators=100),
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(Y, T, X=X)
        cate = model.effect(X)
        results[name] = {"ate": np.mean(cate), "cate_std": np.std(cate)}
    return results
```

## CausalForestDML

```python
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

def fit_causal_forest(X, T, Y, W):
    """Doubly-robust CATE with CIs. X: effect modifiers, W: confounders."""
    est = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=200),
        model_t=GradientBoostingClassifier(n_estimators=200),
        n_estimators=500, min_samples_leaf=5, cv=5, random_state=42,
    )
    est.fit(Y, T, X=X, W=W)
    lb, ub = est.effect_interval(X, alpha=0.05)
    ate_inf = est.ate_inference(X=X)
    return est, est.effect(X), lb, ub, est.feature_importances_
```

## Propensity Scoring with IPW

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

def ipw_ate(X, T, Y, estimator="hajek"):
    """Inverse Probability Weighting with Hajek stabilization."""
    ps_model = LogisticRegression(max_iter=1000, C=0.1)
    ps_model.fit(X, T)
    e = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)
    if estimator == "horvitz_thompson":
        ate = np.mean(T * Y / e) - np.mean((1 - T) * Y / (1 - e))
    elif estimator == "hajek":
        w1, w0 = T / e, (1 - T) / (1 - e)
        ate = np.sum(w1 * Y) / np.sum(w1) - np.sum(w0 * Y) / np.sum(w0)
    return ate, e
```

## Difference-in-Differences

```python
import statsmodels.formula.api as smf

def diff_in_diff(df, time_col, treat_col, outcome_col, pre_periods, post_periods):
    """DiD with parallel trends pre-test."""
    pre_df = df[df[time_col].isin(pre_periods)].copy()
    pre_df["time_numeric"] = pre_df[time_col].rank(method="dense")
    trend_fit = smf.ols(
        f"{outcome_col} ~ {treat_col} * time_numeric", data=pre_df
    ).fit()
    interaction_pval = trend_fit.pvalues[f"{treat_col}:time_numeric"]
    if interaction_pval < 0.05:
        print(f"WARNING: Parallel trends violated (p={interaction_pval:.4f})")
    df = df.copy()
    df["post"] = df[time_col].isin(post_periods).astype(int)
    did_fit = smf.ols(f"{outcome_col} ~ {treat_col} * post", data=df).fit(cov_type="HC1")
    att = did_fit.params[f"{treat_col}:post"]
    return did_fit, att, interaction_pval
```

---

## Probabilistic Programming

PPL frameworks, Bayesian modeling, MCMC inference. Extended patterns in `references/probabilistic-programming.md`.

### PPL Decision Table

| Model Complexity | Speed Need | PPL | Why |
|-----------------|------------|-----|-----|
| Standard regression, hierarchical | Moderate | **PyMC** | Mature API, ArviZ integration |
| Large data, GPU required | High | **NumPyro** | JAX backend, fastest MCMC |
| Deep generative models | High | **Pyro** | PyTorch backend, flexible guides |
| Simple conjugate models | Low | **Stan** (CmdStanPy) | Gold standard HMC |
| Production serving | High | **NumPyro** | JIT-compiled, minimal overhead |
| Time series (structural) | Moderate | **Orbit** / PyMC | Specialized DLM, ETS APIs |
| Gaussian processes | Moderate | **GPyTorch** / PyMC | Scalable exact GPs |

### PyMC Linear Regression

```python
import pymc as pm
import arviz as az

def bayesian_linear_regression(X, y):
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        betas = pm.Normal("betas", mu=0, sigma=5, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=5)
        mu = intercept + pm.math.dot(X, betas)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(
            draws=2000, tune=1000, chains=4,
            target_accept=0.9, random_seed=42,
        )
    return idata
```

### NumPyro NUTS Sampling

```python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)

def numpyro_regression(X, y):
    def model(X, y=None):
        intercept = numpyro.sample("intercept", dist.Normal(0, 10))
        betas = numpyro.sample("betas", dist.Normal(0, 5).expand([X.shape[1]]))
        sigma = numpyro.sample("sigma", dist.HalfNormal(5))
        numpyro.sample("y_obs", dist.Normal(intercept + X @ betas, sigma), obs=y)

    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=4)
    mcmc.run(jax.random.PRNGKey(42), X, y)
    return az.from_numpyro(mcmc)
```

---

## Gotchas

### Causal Inference
- **Propensity score overlap**: Scores near 0/1 cause exploding IPW weights. Always clip (0.01-0.99) and check overlap histograms
- **DoWhy refutations not optional**: Estimate without refutation is meaningless
- **T-Learner bias**: Imbalanced groups cause overfitting on smaller group. Use X-Learner or CausalForestDML
- **DiD parallel trends**: Non-significant pre-trend test does not prove parallel trends. Use multiple pre-periods + visual inspection
- **IV strength**: Weak instruments (first-stage F < 10) cause severe bias. Always report first-stage F-stat
- **CATE vs ATE**: Averaging CATE gives ATE only under correct specification. Report both with CIs
- **Cross-fitting**: DML methods need cross-fitting. Fewer than 3 folds introduces regularization bias

### Probabilistic Programming
- **Non-centered parameterization**: For hierarchical models, use `offset * sigma + mu` not `Normal(mu, sigma)`. Centered causes funnel divergences
- **Divergences are not ignorable**: Even 1 means biased posterior. Increase `target_accept` (0.95-0.99) or reparameterize
- **R-hat must be < 1.01**: Values above 1.05 = chains haven't mixed. Run longer or reparameterize
- **ESS**: Bulk ESS > 400/chain for means, tail ESS > 400 for credible intervals
- **`plot_rank` > `plot_trace`**: Rank plots more reliable for convergence detection
- **PyMC auto-assigns sampler**: Discrete params silently fall back to Metropolis. Use `step=pm.NUTS()` explicitly
- **NumPyro requires JAX arrays**: Pass `jnp.array()` not numpy. Shape mismatches produce cryptic errors
- **LOO > WAIC**: If Pareto k > 0.7, refit with moment matching or use k-fold CV
- **Prior predictive checks**: Always run before fitting. Absurd predictions = priors too wide
