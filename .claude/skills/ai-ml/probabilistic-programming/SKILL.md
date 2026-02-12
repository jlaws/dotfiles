---
name: probabilistic-programming
description: Build Bayesian models with PyMC, NumPyro, and rigorous MCMC diagnostics
---

# Probabilistic Programming

PyMC model patterns, NumPyro GPU inference, prior selection, MCMC diagnostics, posterior predictive checks, variational inference, and model comparison.

## PPL Decision Table

| Model Complexity | Speed Need | PPL | Why |
|-----------------|------------|-----|-----|
| Standard regression, hierarchical | Moderate | **PyMC** | Mature API, ArviZ integration, good defaults |
| Large data, GPU required | High | **NumPyro** | JAX backend, NUTS on GPU, fastest MCMC |
| Deep generative models | High | **Pyro** | PyTorch backend, flexible guide design |
| Simple conjugate models | Low | **Stan** (via CmdStanPy) | Gold standard HMC, extensive diagnostics |
| Educational / prototyping | Low | **PyMC** | Readable syntax, best documentation |
| Production serving | High | **NumPyro** | JIT-compiled, minimal overhead |
| Time series (structural) | Moderate | **Orbit** / PyMC | Specialized APIs for DLM, ETS |
| Gaussian processes | Moderate | **GPyTorch** / PyMC | GPyTorch for scalable exact GPs |

## PyMC Model Patterns

### Linear Regression

```python
import pymc as pm
import numpy as np
import arviz as az

def bayesian_linear_regression(X: np.ndarray, y: np.ndarray) -> az.InferenceData:
    n_features = X.shape[1]
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        betas = pm.Normal("betas", mu=0, sigma=5, shape=n_features)
        sigma = pm.HalfNormal("sigma", sigma=5)
        mu = intercept + pm.math.dot(X, betas)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(
            draws=2000, tune=1000, chains=4,
            target_accept=0.9, random_seed=42,
        )
    return idata
```

### Hierarchical / Multilevel Model

```python
def hierarchical_model(group_idx: np.ndarray, X: np.ndarray,
                       y: np.ndarray, n_groups: int) -> az.InferenceData:
    """Partial pooling with non-centered parameterization."""
    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=5)
        # Non-centered: offset * sigma + mu avoids funnel geometry
        alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, shape=n_groups)
        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset)
        beta = pm.Normal("beta", mu=0, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=5)
        mu = alpha[group_idx] + beta * X
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(2000, tune=1000, chains=4, target_accept=0.95)
    return idata
```

### Gaussian Process

```python
def gp_regression(X: np.ndarray, y: np.ndarray) -> az.InferenceData:
    with pm.Model() as model:
        length_scale = pm.Gamma("length_scale", alpha=2, beta=1)
        amplitude = pm.HalfNormal("amplitude", sigma=2)
        noise = pm.HalfNormal("noise", sigma=1)
        cov = amplitude**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=length_scale)
        gp = pm.gp.Marginal(cov_func=cov)
        gp.marginal_likelihood("y_obs", X=X[:, None], y=y, sigma=noise)
        idata = pm.sample(1000, tune=1000, chains=4)
    return idata
```

## NumPyro for GPU-Accelerated Inference

### NUTS Sampling

```python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

numpyro.set_host_device_count(4)

def numpyro_regression(X: jnp.ndarray, y: jnp.ndarray):
    def model(X, y=None):
        intercept = numpyro.sample("intercept", dist.Normal(0, 10))
        betas = numpyro.sample("betas", dist.Normal(0, 5).expand([X.shape[1]]))
        sigma = numpyro.sample("sigma", dist.HalfNormal(5))
        mu = intercept + X @ betas
        numpyro.sample("y_obs", dist.Normal(mu, sigma), obs=y)

    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=4)
    mcmc.run(jax.random.PRNGKey(42), X, y)
    return az.from_numpyro(mcmc)
```

### Stochastic Variational Inference (SVI)

```python
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

def numpyro_svi(X: jnp.ndarray, y: jnp.ndarray, num_steps: int = 5000):
    """VI for fast approximate posterior when MCMC is too slow."""
    def model(X, y=None):
        intercept = numpyro.sample("intercept", dist.Normal(0, 10))
        betas = numpyro.sample("betas", dist.Normal(0, 5).expand([X.shape[1]]))
        sigma = numpyro.sample("sigma", dist.HalfNormal(5))
        numpyro.sample("y_obs", dist.Normal(intercept + X @ betas, sigma), obs=y)

    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), loss=Trace_ELBO())
    result = svi.run(jax.random.PRNGKey(0), num_steps, X, y)
    predictive = Predictive(guide, params=result.params, num_samples=2000)
    return predictive(jax.random.PRNGKey(1), X, y)
```

## Prior Selection Guide

| Parameter Type | Recommended Prior | Rationale |
|---------------|-------------------|-----------|
| Regression intercept | `Normal(0, 10)` | Weakly informative, centered on zero |
| Regression slope | `Normal(0, 5)` | Allows moderate effects |
| Scale / std dev | `HalfNormal(5)` or `Exponential(1)` | Positive, shrinks toward zero |
| Correlation matrix | `LKJCholesky(eta=2)` | eta=2 weakly favors identity |
| Proportion | `Beta(2, 2)` | Weakly informative, avoids 0/1 edges |
| Count rate | `Gamma(2, 0.5)` | Positive, weakly informative |
| GP length scale | `Gamma(2, 1)` or `InverseGamma(5, 5)` | Prevents near-zero or infinite |
| Degrees of freedom (Student-t) | `Gamma(2, 0.1)` | Allows heavy tails, weakly informative |

### Prior Predictive Workflow

```python
with pm.Model() as model:
    # ... define priors and likelihood ...
    prior_pred = pm.sample_prior_predictive(samples=500)
    # Check: are prior predictions in a plausible range?
    # If y is height in cm, prior predictions of 1e6 cm = priors too wide
    az.plot_ppc(prior_pred, group="prior")
```

## MCMC Diagnostics with ArviZ

```python
import arviz as az

def full_diagnostics(idata: az.InferenceData):
    summary = az.summary(idata, hdi_prob=0.94)
    print(summary)
    rhat = az.rhat(idata)        # all values should be < 1.01
    ess = az.ess(idata)          # bulk ESS > 400 per chain
    divergences = idata.sample_stats.diverging.sum().values
    print(f"Divergences: {divergences}")  # should be 0
    az.plot_trace(idata, var_names=["intercept", "betas", "sigma"])
    az.plot_rank(idata, var_names=["betas"])  # rank plots > trace
    az.plot_energy(idata)
    return summary

def posterior_predictive_check(idata: az.InferenceData, model):
    with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    az.plot_ppc(idata, num_pp_samples=100, kind="cumulative")
    az.plot_loo_pit(idata, y="y_obs")  # should be uniform
```

## Model Comparison

```python
def compare_models(models: dict[str, az.InferenceData]) -> None:
    """Compare via LOO-CV (preferred over WAIC)."""
    for name, idata in models.items():
        loo = az.loo(idata, pointwise=True)
        print(f"{name}: elpd_loo={loo.elpd_loo:.1f} +/- {loo.se:.1f}")
    comparison = az.compare(models, ic="loo")
    print(comparison)
    az.plot_compare(comparison)

# Enable log_likelihood for comparison
# PyMC: pm.sample(..., idata_kwargs={"log_likelihood": True})
# NumPyro: numpyro.infer.log_likelihood(model_fn, mcmc.get_samples(), X, y=y)
```

## Gotchas

- **Non-centered parameterization**: for hierarchical models, use `offset * sigma + mu` not `Normal(mu, sigma)`. Centered causes funnel geometries and divergences.
- **Divergences are not ignorable**: even 1 means biased posterior. Increase `target_accept` (0.95-0.99), reparameterize, or simplify.
- **R-hat must be < 1.01**: values above 1.05 indicate chains haven't mixed. Run longer or reparameterize.
- **ESS**: bulk ESS > 400 per chain for means, tail ESS > 400 for credible intervals. Low tail ESS = unreliable HDI.
- **`plot_rank` > `plot_trace`**: rank plots are more reliable for detecting convergence issues.
- **PyMC auto-assigns sampler**: discrete parameters silently fall back to Metropolis. Use `pm.sample(step=pm.NUTS())` explicitly.
- **NumPyro requires JAX arrays**: pass `jnp.array()` not numpy. Shape mismatches produce cryptic tracing errors.
- **WAIC vs LOO-CV**: prefer LOO. If Pareto k > 0.7, refit with moment matching or use k-fold CV.
- **Prior predictive checks**: always run `pm.sample_prior_predictive()` first. Absurd prior predictions = priors too wide.
