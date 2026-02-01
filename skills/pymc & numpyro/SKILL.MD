---
name: pymc
description: Probabilistic programming for Bayesian statistical modeling and inference. PyMC provides declarative model specification with MCMC (NUTS) and variational inference samplers; NumPyro offers JAX-accelerated equivalent for large-scale problems. Use when: quantifying uncertainty in parameter estimates, building hierarchical or mixed-effects models, Bayesian A/B testing or experimentation, posterior predictive checks, model comparison with WAIC or LOO-CV, scientific measurement with error propagation, any analysis requiring credible intervals, probability statements like P(effect > 0), or situations where understanding the full posterior distribution matters more than a single p-value. Also use when priors encode domain knowledge, sample sizes are small, or data is naturally nested.
version: 5.0.0
license: Apache-2.0
---

# PyMC & NumPyro — Probabilistic Programming

PyMC is the leading Python library for Bayesian statistical modeling. You declare priors and a likelihood — the sampler computes the posterior automatically. NumPyro provides the identical paradigm on JAX for GPU-accelerated inference. Both integrate with ArviZ for diagnostics.

**Core value:** Instead of "is this significant?" (p-value), Bayesian methods answer "what is the full probability distribution over possible values?" — a fundamentally richer answer for every scientific and business question.

## When to Use

- Quantifying uncertainty: credible intervals, prediction intervals, error propagation through calculations.
- Hierarchical models: data naturally nested (students in schools, patients in hospitals, stores in regions).
- A/B testing where "P(B > A) = 0.95" is more actionable than "p = 0.03".
- Scientific measurement where uncertainty must propagate through a pipeline.
- Model comparison: which explanation fits better? (WAIC, LOO-CV)
- Small samples: Bayesian priors regularize and prevent overfitting where MLE fails.
- Any problem where you can articulate the generative story (how data was produced).

**When NOT to use:** Simple hypothesis tests on huge datasets where a p-value suffices. Pure prediction without uncertainty (use sklearn). Real-time inference (MCMC is too slow).

## Reference Documentation

**PyMC docs**: https://pymc.io/en/stable/  
**NumPyro docs**: https://num.pyro.ai/en/stable/  
**ArviZ docs**: https://arviz.org/en/stable/  
**GitHub**: https://github.com/pymc-devs/pymc  
**Search patterns**: `pm.Model`, `pm.sample`, `pm.Normal`, `az.summary`, `az.plot_trace`

## Core Principles

### Bayes' Theorem
**posterior ∝ prior × likelihood**  
Prior = belief before data. Likelihood = how probable is this data given the parameter? Posterior = updated belief after seeing data. PyMC specifies prior + likelihood; MCMC computes the posterior by sampling.

### MCMC — Markov Chain Monte Carlo
The posterior is rarely analytically tractable. MCMC draws samples that, in the limit, are distributed exactly as the posterior. PyMC uses NUTS (No-U-Turn Sampler) — state of the art. Output: thousands of parameter values drawn from the posterior.

### Posterior Predictive Check (PPC)
Generate synthetic data from the fitted model. Compare to real data. If they match visually, the model is reasonable. This catches systematic failures that summary statistics miss — the single most important validation step.

### Credible Intervals
A 95% credible interval [a, b] means: "Given the data, there is a 95% probability the true parameter lies in [a, b]." This is the intuitive interpretation people actually want. (Frequentist confidence intervals do NOT mean this.)

## Quick Reference

### Installation

```bash
pip install pymc arviz            # PyMC + diagnostics
pip install numpyro jax jaxlib    # NumPyro (JAX backend, GPU-accelerated)
```

### Standard Imports

```python
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
```

### Basic Pattern — Coin Flip (Simplest Bayesian Model)

```python
import pymc as pm
import arviz as az

# Observed: 15 heads out of 50 flips
n_flips, n_heads = 50, 15

with pm.Model() as model:
    # Prior: belief about P(heads) before seeing data
    p = pm.Beta('p', alpha=2, beta=2)           # Weakly informative, centered at 0.5

    # Likelihood: how many heads given p
    heads = pm.Binomial('heads', n=n_flips, p=p, observed=n_heads)

    # Sample the posterior
    trace = pm.sample(4000, cores=2, return_inferencedata=True, random_seed=42)

# Interpret posterior
az.summary(trace, var_names=['p'])
# → mean ≈ 0.31, HDI[94%] = [0.19, 0.43]
# Translation: "P(heads) is 31%; 94% credible that it's between 19% and 43%"

az.plot_posterior(trace, var_names=['p'])
```

### Basic Pattern — NumPyro Equivalent

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
import jax.numpy as jnp

n_flips, n_heads = 50, 15

def model():
    p = numpyro.sample('p', dist.Beta(2, 2))
    numpyro.sample('heads', dist.Binomial(n_flips, p), obs=n_heads)

kernel = NUTS(model)
mcmc   = MCMC(kernel, num_warmup=1000, num_samples=4000, num_chains=2)
mcmc.run(random.PRNGKey(0))
samples = mcmc.get_samples()

print(f"P(heads): {samples['p'].mean():.3f} "
      f"HDI=[{jnp.percentile(samples['p'], 3):.3f}, {jnp.percentile(samples['p'], 97):.3f}]")
```

## Critical Rules

### ✅ DO

- **Check convergence ALWAYS** — `az.summary()` shows Rhat. Rhat > 1.01 means not converged. Never trust results without this.
- **Use ≥ 2 chains, prefer 4** — Cannot assess convergence with 1 chain.
- **Do posterior predictive checks** — `pm.sample_posterior_predictive()`. Non-negotiable validation.
- **Use informative priors where possible** — Even weakly informative (`HalfNormal(1)` for σ) beats flat priors. Flat priors on unbounded scales cause convergence failures.
- **Use `HalfNormal` or `Gamma` for scale parameters** — σ must be positive. `pm.HalfNormal('sigma', sigma=1)` is the default choice.
- **Wrap derived quantities in `pm.Deterministic`** — Only way to track computed posteriors (odds ratios, differences, lifts).
- **Set `random_seed`** — Reproducibility.
- **Use `return_inferencedata=True`** — Returns ArviZ InferenceData. Legacy dict format is deprecated.

### ❌ DON'T

- **Don't ignore divergences** — Divergence = sampler lost. Requires reparameterization or prior adjustment.
- **Don't use `Uniform` on scale parameters** — `pm.Uniform('sigma', 0, 100)` causes sampler to struggle near 0. Use `HalfNormal`.
- **Don't confuse `observed` and unobserved** — `observed=data` binds to data. Without it, the variable is sampled from the prior.
- **Don't run only 1 chain** — Cannot diagnose convergence.
- **Don't use PyMC for >100k rows without considering NumPyro** — CPU-bound PyMC becomes very slow. NumPyro + JAX + GPU is 10–100x faster.
- **Don't mix up loss functions / label formats** — In Bayesian context: if labels are integers use `Categorical`; if binary use `Bernoulli`; if proportions use `Beta`.

## Anti-Patterns (NEVER)

```python
import pymc as pm
import numpy as np

# ❌ BAD: Flat prior on scale → sampler struggles near zero
with pm.Model():
    sigma = pm.Uniform('sigma', lower=0, upper=100)  # Divergences guaranteed

# ✅ GOOD: HalfNormal is the standard for positive scales
with pm.Model():
    sigma = pm.HalfNormal('sigma', sigma=1.0)

# ❌ BAD: Ignoring convergence — trusting undiagnosed results
with pm.Model():
    mu  = pm.Normal('mu', mu=0, sigma=10)
    obs = pm.Normal('obs', mu=mu, sigma=1, observed=data)
    trace = pm.sample(100, cores=1)              # Too few draws, 1 chain
print(trace.posterior['mu'].mean().values)        # Meaningless if not converged

# ✅ GOOD: Proper setup + convergence gate
with pm.Model():
    mu  = pm.Normal('mu', mu=0, sigma=10)
    obs = pm.Normal('obs', mu=mu, sigma=1, observed=data)
    trace = pm.sample(4000, cores=4, return_inferencedata=True, random_seed=42)

import arviz as az
summary = az.summary(trace)
assert summary['r_hat'].max() < 1.01, f"Not converged: Rhat={summary['r_hat'].max():.3f}"

# ❌ BAD: Derived quantity not tracked — lost after sampling
with pm.Model():
    p_a = pm.Beta('p_a', 1, 1)
    p_b = pm.Beta('p_b', 1, 1)
    # Difference p_b - p_a is never recorded in the trace

# ✅ GOOD: Deterministic captures derived posteriors
with pm.Model():
    p_a  = pm.Beta('p_a', 1, 1)
    p_b  = pm.Beta('p_b', 1, 1)
    diff = pm.Deterministic('diff', p_b - p_a)    # Now in trace.posterior['diff']

# ❌ BAD: No posterior predictive check — model could be completely wrong
with pm.Model() as model:
    mu    = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    obs   = pm.Normal('obs', mu=mu, sigma=sigma, observed=data)
    trace = pm.sample(4000, return_inferencedata=True)
    # "Converged" ≠ "correct". Convergence just means the sampler found A distribution.

# ✅ GOOD: PPC validates whether the model generates plausible data
with pm.Model() as model:
    mu    = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    obs   = pm.Normal('obs', mu=mu, sigma=sigma, observed=data)
    trace = pm.sample(4000, return_inferencedata=True)
    ppc   = pm.sample_posterior_predictive(trace, model=model)
    az.plot_ppc(ppc, var_names=['obs'])  # Simulated vs real — must look similar
```

## Prior Selection Guide

```
PARAMETER                  | RECOMMENDED PRIOR              | NOTES
---------------------------|--------------------------------|--------------------------------------
Mean / intercept           | Normal(domain_mean, wide_sd)   | Weakly informative; scale to data
Positive scale (σ, τ)      | HalfNormal(1)                  | Default; adjust sigma to data scale
Rate / proportion (0–1)    | Beta(1, 1)                     | Uniform on [0,1]; Beta(2,5) if informed
Count rate (λ)             | Gamma(2, 1) or Exponential(1) | Positive, right-skewed
Regression slope           | Normal(0, 2–5)                 | Weakly regularizing toward zero
Odds ratio                 | Lognormal(0, 1)                | Positive, multiplicative scale
Correlation                | LKJCorrelation(1)              | PyMC-specific; valid [-1, 1]

RULES OF THUMB:
  → Know the rough scale? Use Normal(known_mean, 2–10 × known_std).
  → Know nothing? Wide but bounded priors, NOT Uniform(-∞, ∞).
  → Scale parameters (σ, τ) ALWAYS positive → HalfNormal, Gamma, or Exponential.
  → More data = prior matters less. With n > 100, even bad priors get washed out.
```

## Model Specification — PyMC Patterns

### Basic Structure

```python
import pymc as pm

with pm.Model() as model:
    # 1. PRIORS — parameters to learn
    mu    = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # 2. LIKELIHOOD — data generation mechanism
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=data)

    # 3. DERIVED QUANTITIES (optional)
    # pm.Deterministic('name', expression)

    # 4. SAMPLE
    trace = pm.sample(4000, cores=4, return_inferencedata=True, random_seed=42)
```

### Linear Regression

```python
import pymc as pm
import numpy as np

# X shape: (n,) single predictor or (n, p) multiple
# y shape: (n,)

with pm.Model():
    intercept = pm.Normal('intercept', mu=0, sigma=5)
    slope     = pm.Normal('slope', mu=0, sigma=5)           # shape=(p,) for multivariate
    sigma     = pm.HalfNormal('sigma', sigma=2)

    mu    = pm.Deterministic('mu', intercept + slope * X)   # Track mean in trace
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(4000, cores=4, return_inferencedata=True, random_seed=42)
```

### Multivariate Regression

```python
import pymc as pm
import numpy as np

X = np.array([...])   # Shape: (n, p)
y = np.array([...])
n, p = X.shape

with pm.Model():
    intercept = pm.Normal('intercept', mu=0, sigma=5)
    slopes    = pm.Normal('slopes', mu=0, sigma=5, shape=p)    # One per feature
    sigma     = pm.HalfNormal('sigma', sigma=2)

    mu    = pm.Deterministic('mu', intercept + X @ slopes)     # Matrix multiply
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(4000, cores=4, return_inferencedata=True)
```

## Sampling and Diagnostics

### Sampling Parameters

```python
import pymc as pm

with pm.Model() as model:
    # ... model ...
    trace = pm.sample(
        draws=4000,                    # Posterior samples per chain (after warmup)
        tune=2000,                     # Warmup draws — discarded; sampler adapts here
        cores=4,                       # Parallel chains = CPU cores
        chains=4,                      # Independent chains for convergence check
        return_inferencedata=True,     # ArviZ InferenceData format
        random_seed=42,                # Reproducibility
    )
```

### Diagnostics Checklist

```python
import pymc as pm
import arviz as az

# 1. SUMMARY — Rhat + ESS at a glance
#    Rhat ≈ 1.00 → converged | ESS > 100 → enough effective samples
summary = az.summary(trace)
print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk']])

# 2. TRACE PLOT — visual convergence
#    Chains should overlap, look like random noise. No trends or drift.
az.plot_trace(trace, var_names=['mu', 'sigma'])

# 3. POSTERIOR
az.plot_posterior(trace, var_names=['mu', 'sigma'])

# 4. ENERGY — divergence check
az.plot_energy(trace)   # Two distributions should overlap heavily

# 5. PPC — model validation
ppc = pm.sample_posterior_predictive(trace, model=model)
az.plot_ppc(ppc, var_names=['y_obs'])

# ─── Convergence decision tree ───
# Rhat > 1.05  → STOP. Increase tune, add chains, or reparameterize.
# Rhat 1.01–1.05 → WARNING. Increase draws and tune.
# Rhat < 1.01  → OK. Verify ESS > 100.
# Many divergences → Reparameterize or tighten priors.
```

## NumPyro — The Fast Alternative

### When to Switch

```
PyMC (default):
  ✅ Easier syntax, better diagnostics, richer tutorials
  ✅ Use when: accuracy > speed, data < 50k rows

NumPyro (switch when):
  ✅ JAX backend → GPU-accelerated, 10–100x faster
  ✅ Use when: data > 50k rows, you have a GPU, or need speed
```

### Side-by-Side: Linear Regression

```python
# ─── PyMC ─────────────────────────────────────────────
import pymc as pm

with pm.Model():
    intercept = pm.Normal('intercept', mu=0, sigma=5)
    slope     = pm.Normal('slope', mu=0, sigma=5)
    sigma     = pm.HalfNormal('sigma', sigma=2)
    mu        = intercept + slope * X
    y_obs     = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    trace     = pm.sample(4000, cores=4, return_inferencedata=True)

# ─── NumPyro equivalent ────────────────────────────────
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random

def model(X, y=None):
    intercept = numpyro.sample('intercept', dist.Normal(0, 5))
    slope     = numpyro.sample('slope',     dist.Normal(0, 5))
    sigma     = numpyro.sample('sigma',     dist.HalfNormal(2))
    mu        = intercept + slope * X
    numpyro.sample('y', dist.Normal(mu, sigma), obs=y)

kernel = NUTS(model)
mcmc   = MCMC(kernel, num_warmup=2000, num_samples=4000, num_chains=4)
mcmc.run(random.PRNGKey(0), X=X_train, y=y_train)
samples = mcmc.get_samples()

# Prediction: full posterior predictive
pred_fn  = Predictive(model, samples)
preds    = pred_fn(random.PRNGKey(1), X=X_test)
# preds['y'] shape: (4000, n_test) — one prediction per posterior sample
```

### NumPyro: Multivariate

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
import jax.numpy as jnp

def model(X, y=None):
    n, p    = X.shape
    slopes    = numpyro.sample('slopes',    dist.Normal(jnp.zeros(p), 5))
    intercept = numpyro.sample('intercept', dist.Normal(0, 5))
    sigma     = numpyro.sample('sigma',     dist.HalfNormal(2))
    mu        = intercept + X @ slopes
    numpyro.sample('y', dist.Normal(mu, sigma), obs=y)

kernel = NUTS(model)
mcmc   = MCMC(kernel, num_warmup=2000, num_samples=4000, num_chains=4)
mcmc.run(random.PRNGKey(42), X=X_train, y=y_train)
```

## Common Model Patterns

### 1. Linear Regression with Uncertainty Bands

```python
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.randn(100)
y = 2.5 * X + 1.0 + np.random.randn(100) * 0.8

with pm.Model():
    intercept = pm.Normal('intercept', mu=0, sigma=5)
    slope     = pm.Normal('slope', mu=0, sigma=5)
    sigma     = pm.HalfNormal('sigma', sigma=2)
    mu        = pm.Deterministic('mu', intercept + slope * X)
    y_obs     = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    trace     = pm.sample(4000, cores=4, return_inferencedata=True, random_seed=42)

# Posterior of slope (compare to true value 2.5)
az.plot_posterior(trace, var_names=['slope'], ref_val=2.5)

# Prediction with credible + predictive bands
X_new      = np.linspace(-3, 3, 100)
slopes     = trace.posterior['slope'].values.flatten()
intercepts = trace.posterior['intercept'].values.flatten()
sigmas     = trace.posterior['sigma'].values.flatten()

mu_pred = intercepts[:, None] + slopes[:, None] * X_new[None, :]       # (samples, 100)
y_pred  = mu_pred + sigmas[:, None] * np.random.randn(*mu_pred.shape)  # Add obs noise

plt.fill_between(X_new, np.percentile(y_pred, 2.5, 0),  np.percentile(y_pred, 97.5, 0),
                 alpha=0.15, color='blue', label='95% predictive interval')
plt.fill_between(X_new, np.percentile(mu_pred, 2.5, 0), np.percentile(mu_pred, 97.5, 0),
                 alpha=0.35, color='blue', label='95% credible interval (mean)')
plt.plot(X_new, mu_pred.mean(0), 'k-', lw=2, label='Posterior mean')
plt.scatter(X, y, s=15, alpha=0.5, color='gray', label='Data')
plt.legend(); plt.xlabel('X'); plt.ylabel('y'); plt.show()
```

### 2. Hierarchical Model — Borrowing Strength Across Groups

```python
import pymc as pm
import arviz as az
import numpy as np

# Test scores: students nested in schools, schools have different sizes
n_schools          = 8
students_per_school = np.array([5, 3, 12, 8, 2, 15, 6, 4])
school_ids         = np.repeat(np.arange(n_schools), students_per_school)
scores             = np.array([...])   # All student scores

with pm.Model():
    # Population-level hyperparameters
    mu_pop  = pm.Normal('mu_pop', mu=60, sigma=20)        # Overall mean
    tau_pop = pm.HalfNormal('tau_pop', sigma=10)           # Between-school variation

    # School-level means — drawn from population (the hierarchical link)
    school_mu = pm.Normal('school_mu', mu=mu_pop, sigma=tau_pop, shape=n_schools)

    # Within-school noise
    sigma = pm.HalfNormal('sigma', sigma=5)

    # Each student's score depends on their school
    score = pm.Normal('score', mu=school_mu[school_ids], sigma=sigma, observed=scores)

    trace = pm.sample(4000, cores=4, return_inferencedata=True, random_seed=42)

# Key effect: small schools (n=2, 3) SHRINK toward population mean
# Large schools (n=12, 15) keep their own signal
# This is automatic Bayesian regularization — no manual tuning needed
az.plot_forest(trace, var_names=['school_mu'], combined=True)
```

### 3. Bayesian A/B Test

```python
import pymc as pm
import arviz as az
import numpy as np

# Conversion data
n_control,   conv_control   = 1000, 45   # 4.5%
n_treatment, conv_treatment = 1200, 72   # 6.0%

with pm.Model():
    # Priors: conversion rates (Beta is natural for proportions)
    p_control   = pm.Beta('p_control',   alpha=1, beta=1)
    p_treatment = pm.Beta('p_treatment', alpha=1, beta=1)

    # Likelihoods: observed conversions
    pm.Binomial('obs_control',   n=n_control,   p=p_control,   observed=conv_control)
    pm.Binomial('obs_treatment', n=n_treatment, p=p_treatment, observed=conv_treatment)

    # Derived quantities — the decision metrics
    diff = pm.Deterministic('diff', p_treatment - p_control)
    lift = pm.Deterministic('lift', (p_treatment - p_control) / p_control)

    trace = pm.sample(10000, cores=4, return_inferencedata=True, random_seed=42)

# Decision outputs
diff_post = trace.posterior['diff'].values.flatten()
lift_post = trace.posterior['lift'].values.flatten()

p_better      = (diff_post > 0).mean()
p_lift_gt_10  = (lift_post > 0.10).mean()       # P(lift > 10%)
expected_lift = lift_post.mean()

print(f"P(treatment > control):  {p_better:.3f}")
print(f"P(lift > 10%):           {p_lift_gt_10:.3f}")
print(f"Expected lift:           {expected_lift:.1%}")
print(f"Lift HDI [3%, 97%]:      [{np.percentile(lift_post, 3):.1%}, {np.percentile(lift_post, 97):.1%}]")

# Ship rule: P(lift > MDE) > decision_threshold → ship
# e.g., if P(lift > 5%) > 0.90 → ship
```

### 4. Logistic Regression

```python
import pymc as pm
import arviz as az
import numpy as np

X = np.array([...])   # Shape: (n, p)
y = np.array([...])   # Binary: 0 or 1

with pm.Model():
    intercept = pm.Normal('intercept', mu=0, sigma=2)
    slopes    = pm.Normal('slopes', mu=0, sigma=2, shape=X.shape[1])

    # Log-odds → probability via sigmoid
    logit_p = intercept + X @ slopes
    p       = pm.Deterministic('p', pm.math.sigmoid(logit_p))

    # Likelihood
    y_obs = pm.Bernoulli('y_obs', p=p, observed=y)

    trace = pm.sample(4000, cores=4, return_inferencedata=True, random_seed=42)

# Odds ratios with credible intervals
import arviz as az
slopes_post = trace.posterior['slopes'].values   # (chains, draws, p)
odds_ratios = np.exp(slopes_post)

for i in range(X.shape[1]):
    or_vals = odds_ratios[:, :, i].flatten()
    print(f"Feature {i}: OR = {or_vals.mean():.2f} "
          f"HDI=[{np.percentile(or_vals, 3):.2f}, {np.percentile(or_vals, 97):.2f}]")
    # If HDI excludes 1.0 → feature has credible effect on odds
```

### 5. Poisson Regression (Count Data)

```python
import pymc as pm
import numpy as np

X = np.array([...])   # Covariates, shape: (n, p)
y = np.array([...])   # Counts: 0, 1, 2, 3, ... (non-negative integers)

with pm.Model():
    intercept = pm.Normal('intercept', mu=0, sigma=2)
    slopes    = pm.Normal('slopes', mu=0, sigma=2, shape=X.shape[1])

    # Log-rate link: λ = exp(Xβ)
    log_mu = intercept + X @ slopes
    mu     = pm.Deterministic('mu', pm.math.exp(log_mu))

    y_obs = pm.Poisson('y_obs', mu=mu, observed=y)

    trace = pm.sample(4000, cores=4, return_inferencedata=True, random_seed=42)

# Rate ratios: exponentiated slopes = multiplicative effect on expected count
rate_ratios = np.exp(trace.posterior['slopes'].values)
for i in range(X.shape[1]):
    rr = rate_ratios[:, :, i].flatten()
    print(f"Feature {i}: RR = {rr.mean():.2f} HDI=[{np.percentile(rr, 3):.2f}, {np.percentile(rr, 97):.2f}]")
```

### 6. AR(1) Time Series

```python
import pymc as pm
import arviz as az
import numpy as np

y = np.array([...])   # Observed time series, shape: (T,)

with pm.Model():
    mu    = pm.Normal('mu', mu=y.mean(), sigma=y.std() * 3)   # Long-term mean
    rho   = pm.Uniform('rho', lower=-0.99, upper=0.99)        # Autocorrelation
    sigma = pm.HalfNormal('sigma', sigma=y.std())              # Innovation noise

    # Conditional likelihood: y_t | y_{t-1} ~ Normal(mu + rho*(y_{t-1} - mu), sigma)
    # For observed series, previous values are known — vectorized, no loop needed
    y_lag     = y[:-1]                                         # y_{t-1} (known)
    expected  = mu + rho * (y_lag - mu)
    y_obs     = pm.Normal('y_obs', mu=expected, sigma=sigma, observed=y[1:])

    trace = pm.sample(4000, cores=4, return_inferencedata=True, random_seed=42)

# Posterior of autocorrelation coefficient
az.plot_posterior(trace, var_names=['rho', 'mu', 'sigma'])
# If rho HDI excludes 0 → significant autocorrelation
```

## Model Comparison

```python
import pymc as pm
import arviz as az
import numpy as np

# ─── Fit two competing models ───

# Model A: intercept only
with pm.Model() as model_a:
    mu    = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=2)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    trace_a = pm.sample(4000, cores=4, return_inferencedata=True)
    pm.sample_posterior_predictive(trace_a, extend_inferencedata=True)

# Model B: with covariate
with pm.Model() as model_b:
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    slope     = pm.Normal('slope', mu=0, sigma=5)
    sigma     = pm.HalfNormal('sigma', sigma=2)
    mu        = intercept + slope * X
    y_obs     = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    trace_b   = pm.sample(4000, cores=4, return_inferencedata=True)
    pm.sample_posterior_predictive(trace_b, extend_inferencedata=True)

# ─── Compare via WAIC or LOO ───
comparison = az.compare(
    {'intercept_only': trace_a, 'with_covariate': trace_b},
    ic='loo'        # or ic='waic'
)
print(comparison[['loo', 'dloo', 'dloo_se', 'weight']])
# Lower LOO = better. If |dloo| > 2 * dloo_se → meaningfully different.
# 'weight' column: model averaging weights (sum to 1)

az.plot_compare(comparison)

# ─── Interpretation ───
# LOO/WAIC automatically penalize complexity (like AIC/BIC but Bayesian).
# Best model: comparison.index[0]
# If weights are 0.9 vs 0.1 → clear winner.
# If weights are 0.6 vs 0.4 → both plausible, consider model averaging.
```

## Practical Workflows

### 1. A/B Test with Full Decision Support

```python
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bayesian_ab_test(data: pd.DataFrame,
                     group_col: str,
                     conversion_col: str,
                     min_detectable_effect: float = 0.05,
                     decision_threshold: float = 0.90):
    """
    Full Bayesian A/B test → ship / don't ship decision.
    Returns probability treatment wins, expected lift, confidence in exceeding MDE.
    """
    groups = data[group_col].unique()
    assert len(groups) == 2

    stats = {}
    for g in groups:
        mask = data[group_col] == g
        stats[g] = {'n': int(mask.sum()), 'conv': int(data.loc[mask, conversion_col].sum())}

    ctrl, treat = groups[0], groups[1]

    with pm.Model():
        p_c = pm.Beta('p_control',   alpha=1, beta=1)
        p_t = pm.Beta('p_treatment', alpha=1, beta=1)
        pm.Binomial('obs_c', n=stats[ctrl]['n'],  p=p_c, observed=stats[ctrl]['conv'])
        pm.Binomial('obs_t', n=stats[treat]['n'], p=p_t, observed=stats[treat]['conv'])
        pm.Deterministic('diff', p_t - p_c)
        pm.Deterministic('lift', (p_t - p_c) / p_c)
        trace = pm.sample(10000, cores=4, return_inferencedata=True, random_seed=42)

    diff_post = trace.posterior['diff'].values.flatten()
    lift_post = trace.posterior['lift'].values.flatten()

    p_better      = (diff_post > 0).mean()
    p_lift_gt_mde = (lift_post > min_detectable_effect).mean()
    decision      = "✅ SHIP" if p_lift_gt_mde >= decision_threshold else "❌ DON'T SHIP"

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.hist(lift_post, bins=120, density=True, alpha=0.7, color='steelblue', edgecolor='none')
    ax1.axvline(0, color='red', ls='--', lw=1.5, label='No effect')
    ax1.axvline(min_detectable_effect, color='green', ls='--', lw=1.5, label=f'MDE = {min_detectable_effect:.0%}')
    ax1.set_xlabel('Relative lift')
    ax1.set_title(f'{decision}\nP(lift > MDE) = {p_lift_gt_mde:.1%}')
    ax1.legend(fontsize=9)

    ax2.hist(diff_post, bins=120, density=True, alpha=0.7, color='coral', edgecolor='none')
    ax2.axvline(0, color='red', ls='--', lw=1.5)
    ax2.set_xlabel('Absolute difference')
    ax2.set_title(f'P(treatment > control) = {p_better:.1%}')
    plt.tight_layout(); plt.show()

    return {
        'p_treatment_better': p_better,
        'p_lift_gt_mde':      p_lift_gt_mde,
        'expected_lift':      lift_post.mean(),
        'lift_hdi_94':        (np.percentile(lift_post, 3), np.percentile(lift_post, 97)),
        'decision':           decision
    }

# report = bayesian_ab_test(df, group_col='variant', conversion_col='purchased')
```

### 2. Hierarchical Geo-Marketing Test

```python
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd

def hierarchical_geo_test(df: pd.DataFrame,
                          region_col: str,
                          treatment_col: str,
                          outcome_col: str):
    """
    Regions vary in baseline; treatment effect is shared across all regions.
    Small regions borrow strength from the population — no region left noisy.
    """
    regions  = df[region_col].cat.codes.values
    n_reg    = df[region_col].nunique()
    treated  = df[treatment_col].values.astype(float)
    outcome  = df[outcome_col].values.astype(float)

    with pm.Model():
        # Population baseline
        mu_base    = pm.Normal('mu_baseline', mu=outcome.mean(), sigma=outcome.std())
        tau_region = pm.HalfNormal('tau_region', sigma=1.0)

        # Per-region baselines — shrink toward population
        region_eff = pm.Normal('region_effect', mu=0, sigma=tau_region, shape=n_reg)

        # THE quantity of interest: shared treatment effect
        treatment_eff = pm.Normal('treatment_effect', mu=0, sigma=outcome.std())

        # Noise
        sigma = pm.HalfNormal('sigma', sigma=outcome.std())

        # Expected outcome
        mu = mu_base + region_eff[regions] + treatment_eff * treated
        pm.Normal('y', mu=mu, sigma=sigma, observed=outcome)

        trace = pm.sample(4000, cores=4, return_inferencedata=True, random_seed=42)

    # Key output: treatment effect with full uncertainty
    effect = trace.posterior['treatment_effect'].values.flatten()
    p_pos  = (effect > 0).mean()

    print(f"Treatment effect: {effect.mean():.3f} ± {effect.std():.3f}")
    print(f"P(effect > 0):    {p_pos:.3f}")
    print(f"94% HDI:          [{np.percentile(effect, 3):.3f}, {np.percentile(effect, 97):.3f}]")

    az.plot_posterior(trace, var_names=['treatment_effect'], ref_val=0)
    return trace

# trace = hierarchical_geo_test(df, 'region', 'is_treated', 'revenue')
```

### 3. Scientific Measurement — Error Propagation

```python
import pymc as pm
import arviz as az
import numpy as np

def calibrate_instrument(readings, measurements, new_readings):
    """
    Calibration: true_value = a * reading + b + noise.
    Predict new readings with FULL uncertainty propagation:
    both parameter uncertainty AND measurement noise.
    """
    readings     = np.array(readings, dtype=float)
    measurements = np.array(measurements, dtype=float)
    new_readings = np.array(new_readings, dtype=float)

    with pm.Model() as model:
        # Calibration curve parameters
        a     = pm.Normal('a', mu=1.0, sigma=2.0)          # Scale
        b     = pm.Normal('b', mu=0.0, sigma=1.0)          # Offset
        sigma = pm.HalfNormal('sigma', sigma=0.5)           # Measurement noise

        # Calibration likelihood
        mu_cal = a * readings + b
        pm.Normal('calibration', mu=mu_cal, sigma=sigma, observed=measurements)

        # Prediction: deterministic mean for new readings
        mu_new = pm.Deterministic('predicted_mean', a * new_readings + b)

        trace = pm.sample(4000, cores=4, return_inferencedata=True, random_seed=42)

    # Posterior predictive for new readings (includes measurement noise)
    # Sample: predicted_value = a*x + b + sigma*eps
    a_post     = trace.posterior['a'].values.flatten()
    b_post     = trace.posterior['b'].values.flatten()
    sigma_post = trace.posterior['sigma'].values.flatten()

    for i, x_new in enumerate(new_readings):
        mu_samples   = a_post * x_new + b_post
        pred_samples = mu_samples + sigma_post * np.random.randn(len(sigma_post))

        print(f"Reading {x_new:.1f} → "
              f"Predicted: {pred_samples.mean():.3f} ± {pred_samples.std():.3f} | "
              f"94% HDI: [{np.percentile(pred_samples, 3):.3f}, {np.percentile(pred_samples, 97):.3f}]")

    return trace

# trace = calibrate_instrument(
#     readings=[1, 2, 3, 4, 5],
#     measurements=[2.1, 4.0, 5.9, 8.1, 10.0],
#     new_readings=[6.0, 7.0]
# )
```

### 4. Bayesian Model Selection Pipeline

```python
import pymc as pm
import arviz as az
import numpy as np

def compare_regression_models(X, y, feature_sets: dict):
    """
    Compare regression models defined by different feature subsets.

    feature_sets: dict of {name: list_of_column_indices}
    e.g., {'baseline': [0], 'add_age': [0,1], 'full': [0,1,2,3]}

    Returns: comparison table + traces + best model name
    """
    traces = {}

    for name, cols in feature_sets.items():
        X_sub = X[:, cols]
        p     = len(cols)

        with pm.Model():
            intercept = pm.Normal('intercept', mu=0, sigma=10)
            slopes    = pm.Normal('slopes', mu=0, sigma=5, shape=p)
            sigma     = pm.HalfNormal('sigma', sigma=y.std())
            mu        = intercept + X_sub @ slopes
            pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

            trace = pm.sample(4000, cores=4, return_inferencedata=True, random_seed=42)
            pm.sample_posterior_predictive(trace, extend_inferencedata=True)
            traces[name] = trace
            print(f"  ✓ {name} sampled")

    # Compare
    comparison = az.compare(traces, ic='loo')
    print("\n" + comparison[['loo', 'dloo', 'dloo_se', 'weight']].to_string())
    az.plot_compare(comparison)

    best  = comparison.index[0]
    print(f"\nBest model: {best} (LOO weight = {comparison.loc[best, 'weight']:.2f})")
    return comparison, traces, best

# feature_sets = {
#     'intercept_only': [0],
#     'linear':         [0, 1],
#     'quadratic':      [0, 1, 2],
#     'full':           [0, 1, 2, 3],
# }
# comp, traces, best = compare_regression_models(X, y, feature_sets)
```

## Performance and Scaling

### PyMC vs NumPyro Decision

```
Data < 10k rows, simple model      → PyMC + MCMC. Best diagnostics.
Data 10k–100k rows                 → PyMC first. If >30 min → switch to NumPyro.
Data > 100k rows or GPU available  → NumPyro + JAX directly.
Need speed >> accuracy             → Variational inference (see below).
```

### Variational Inference — Fast Approximation

```python
import pymc as pm
import arviz as az

with pm.Model() as model:
    mu    = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=2)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # ADVI: approximate posterior with a Gaussian. Much faster than MCMC.
    approx = pm.fit(method='advi', n=10000)

    # Draw from approximate posterior
    trace_vi = approx.sample(5000)

# Validate: run MCMC on a data subset, compare posteriors.
# If they agree → safe to use VI on full data.
```

### Parallel Chains

```python
import pymc as pm
import os

n_cores = os.cpu_count()

with pm.Model():
    # ... model ...
    trace = pm.sample(
        draws=4000, tune=2000,
        cores=n_cores,    # One chain per core
        chains=n_cores,   # Match chains to cores for max parallelism
        return_inferencedata=True
    )
```

## Common Pitfalls and Solutions

### Divergences

```python
# SYMPTOM: "X divergences detected" in sampling output
# MEANING: Sampler got lost in a region of high curvature

# Diagnose:
import arviz as az
az.plot_energy(trace)   # Energy-4 should overlap with marginal energy

# FIX (in order of likelihood):
# 1. Reparameterize — use log-scale for positive parameters:
#    Instead of: sigma = pm.HalfNormal('sigma', sigma=1)
#    Try:        log_sigma = pm.Normal('log_sigma', mu=0, sigma=1)
#                sigma = pm.Deterministic('sigma', pm.math.exp(log_sigma))
#
# 2. Increase warmup: pm.sample(tune=5000)
# 3. Increase target_accept: pm.sample(target_accept=0.95)  — more conservative steps
# 4. Tighten priors — reduces the volume the sampler must explore
```

### Slow Sampling

```python
# Benchmark: how long per 100 draws?
import time

with pm.Model() as model:
    # ... model ...
    start = time.time()
    _ = pm.sample(100, tune=100, cores=1, chains=1)
    per_100 = time.time() - start
    print(f"100 draws: {per_100:.1f}s → full run estimate: {per_100 * 40 + per_100 * 20:.0f}s")

# If per_100 > 10s:
#   → Switch to NumPyro (JAX + GPU)
#   → Or use variational inference: pm.fit(method='advi')
#   → Or simplify the model (fewer parameters, simpler likelihood)
```

### Model Converges But PPC Fails

```python
# SYMPTOM: Rhat OK, but simulated data looks nothing like real data
# MEANING: The model converged to the WRONG distribution

# Diagnose:
ppc = pm.sample_posterior_predictive(trace, model=model)
az.plot_ppc(ppc)   # Obvious mismatch

# FIX:
# 1. Check the generative story — does the likelihood match how data was actually produced?
# 2. Add missing structure: interactions, nonlinear terms, extra variables
# 3. Switch likelihood for heavy tails: Normal → StudentT
# 4. Check outliers: plot residuals, identify systematic bias
# 5. Visualize per-observation: predicted vs observed — find WHERE it breaks
```

### Label Switching in Mixture Models

```python
# SYMPTOM: Rhat looks bad for mixture components, but chains mirror each other
# MEANING: Component labels swap between chains (mu_1=0,mu_2=5 vs mu_1=5,mu_2=0)

# FIX: ordered constraint forces mu_1 < mu_2
with pm.Model():
    mu = pm.Uniform('mu', lower=-10, upper=10, shape=2,
                    transform=pm.distributions.transforms.ordered)
    # Now mu[0] < mu[1] guaranteed → no label switching
```

---

PyMC and NumPyro shift every question from "is this significant?" to "what is the full distribution of possibilities, and how confident are we?" The workflow is always: **specify the generative story → sample the posterior → check convergence → validate with PPC → interpret**. Master this loop and Bayesian inference becomes a universal tool across every scientific and business domain.