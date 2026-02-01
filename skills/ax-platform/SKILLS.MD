---
name: ax-platform
description: Adaptive Experimentation Platform (Meta) for Bayesian optimization and scientific experiment design. Ax closes the loop on experiments: define a search space, run trials, observe results, and let a Bayesian surrogate model decide what to try next. Covers single-objective optimization, multi-objective Pareto optimization, outcome constraints, noisy evaluations with uncertainty, and the full ask-tell loop. Use when: designing experiments to find optimal parameters efficiently (fewer trials than grid/random search), running Bayesian optimization (BO) over continuous or discrete search spaces, optimizing multiple competing objectives simultaneously (Pareto frontier), incorporating constraints into optimization (e.g., "latency must stay under 100ms"), handling noisy/repeated measurements where observations have uncertainty, automating the design-run-analyze-repeat cycle in scientific or engineering experiments, or any task where the question is "what should I try next to reach my goal fastest?" Complements Optuna (lighter-weight HPO) — use Ax when you need the full experiment management loop, multi-objective Pareto, or outcome constraints. Complements PyMC (uncertainty quantification on results) and DoWhy (causal interpretation of effects).
version: 0.4.x
license: MIT
---

# Ax — Adaptive Experimentation Platform

Ax answers the single most important question in experimental science: **"What should I try next?"** Instead of exhaustive grid search or blind random search, Ax builds a probabilistic surrogate model (Gaussian Process) of your objective function, then uses that model to suggest the next experiment that is most likely to find the optimum. Fewer trials. Smarter choices. The same paradigm that drives drug discovery, materials science, and ML hyperparameter tuning at Meta.

## Core Mental Model

```
THE ADAPTIVE LOOP:

  ┌─────────────────────────────────────────────┐
  │  1. EXPLORE (Sobol phase)                   │
  │     Quasi-random space-filling design.       │
  │     Builds initial dataset for surrogate.    │
  │     Default: ~5 trials or 2×num_parameters   │
  └──────────────┬──────────────────────────────┘
                 ↓
  ┌─────────────────────────────────────────────┐
  │  2. EXPLOIT (Bayesian Optimization phase)   │
  │     Fit GP surrogate to observed data.       │
  │     Acquisition function picks next trial:   │
  │       • Expected Improvement                 │
  │       • Thompson Sampling                    │
  │     Balances exploration vs exploitation.    │
  └──────────────┬──────────────────────────────┘
                 ↓
  ┌─────────────────────────────────────────────┐
  │  3. ASK → RUN → TELL                        │
  │     ask:  get suggested parameters           │
  │     run:  execute YOUR experiment            │
  │     tell: report the metric back to Ax       │
  │     Repeat until budget exhausted.           │
  └─────────────────────────────────────────────┘

WHY BAYESIAN OVER RANDOM?
  Random search: N trials → N random points in space.
  Bayesian:      N trials → N strategically chosen points.
  Same budget, but BO finds optima 5–50× faster on typical functions.
  Advantage grows with dimensionality and budget constraint.

WHEN BO STRUGGLES:
  • > 20 parameters (curse of dimensionality for GP)
  • Discontinuous / highly multimodal landscapes
  • Extremely cheap evaluations (just use random — overhead not worth it)
  Solution: Ax auto-falls back to Sobol for >20 params.
```

## Ax vs Optuna — Decision Matrix

```
USE Ax WHEN:                                USE Optuna WHEN:
────────────────────────────────            ────────────────────────────
Multi-objective (Pareto frontier)           Single-objective HPO only
Outcome constraints ("latency < 100ms")     Pure speed of iteration matters
Experiment tracking matters                 Integration with ML frameworks
Scientific experiment design                Pruning of unpromising trials
Noisy measurements with known SEM           Lightweight, minimal overhead
Full design → analyze → next cycle          Simple black-box optimization
```

## Reference Documentation

**Ax docs**: https://ax.dev/docs/  
**Tutorials**: https://ax.dev/docs/tutorials.html  
**GitHub**: https://github.com/facebook/ax  
**Search patterns**: `AxClient`, `create_experiment`, `get_next_trial`, `complete_trial`, `ObjectiveProperties`

## Core Principles

### Experiment = Container for All Trials
An Experiment holds the search space definition, all trial results, and the optimization history. One experiment = one optimization run. You can save and resume experiments.

### Trial = Single Evaluation
Each trial is one run of your experiment at specific parameter values. Ax assigns trial indices automatically. A trial can be CANDIDATE → RUNNING → COMPLETED (or FAILED).

### Ask-Tell = The User's Interface
Ax doesn't run your experiments. It suggests parameters (ask), you run whatever black-box process you want (run), and you report the metric back (tell). This keeps Ax decoupled from your simulation/lab/training code.

### Surrogate Model = Gaussian Process
Ax fits a GP to observed (parameters → metric) pairs. The GP predicts the metric at any point in the search space AND gives a confidence interval. The acquisition function uses this uncertainty to decide where to look next.

### Sobol Initialization Is Mandatory
GP needs initial data to fit. The first N trials use Sobol sequences (low-discrepancy quasi-random) to cover the space efficiently. Ax handles this transition automatically — you don't need to manage it.

## Quick Reference

### Installation

```bash
pip install ax-platform
# For Bayesian optimization (required):
pip install botorch gpytorch
# Full install with plotting:
pip install ax-platform[plotting]
```

### Standard Imports

```python
from ax.service.ax_client import AxClient, ObjectiveProperties
import numpy as np
```

### Basic Pattern — Single-Objective Optimization

```python
from ax.service.ax_client import AxClient

# 1. Create client and experiment
ax_client = AxClient()
ax_client.create_experiment(
    name="my_optimization",
    parameters=[
        {"name": "x", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "y", "type": "range", "bounds": [-5.0, 5.0]},
    ],
    objective_name="metric",
    minimize=True
)

# 2. Ask-Tell loop
def black_box(x, y):
    """Your experiment / simulation / training run."""
    return (x - 2)**2 + (y + 3)**2 + 0.1 * np.random.randn()  # Noisy optimum at (2, -3)

for i in range(25):
    parameters, trial_index = ax_client.get_next_trial()
    # Run your experiment
    result = black_box(**parameters)
    # Report back
    ax_client.complete_trial(trial_index=trial_index, raw_data={"metric": result})

# 3. Get the best
best_params, (best_mean, best_sem) = ax_client.get_best_parameters()
print(f"Best parameters: {best_params}")
print(f"Best metric:     {best_mean:.4f} ± {best_sem:.4f}")
```

## Critical Rules

### ✅ DO

- **Start with 20–30 trials minimum** — Sobol phase needs ~5–10 trials; BO needs at least 10–15 more to converge. Budget of 20–30 is the minimum for meaningful optimization.
- **Use `log_scale: True` for parameters that span orders of magnitude** — Learning rates (1e-5 to 1e-1), regularization strengths, etc. Without log scale, BO wastes budget on the wrong region.
- **Report SEM when measurements are noisy** — Pass `(mean, sem)` tuple. Ax uses this uncertainty to avoid chasing noise. Without it, BO overfits to noisy observations.
- **Use `ChoiceParameter` for truly discrete options** — Optimizers like "adam" vs "sgd" are not on a continuous scale. Use `"type": "choice"`, not range.
- **Run multiple seeds / repetitions for noisy objectives** — Average them before reporting, or report the mean + SEM. Single noisy evaluations mislead the surrogate.
- **Save the experiment periodically** — Use `ax_client.save_experiment()`. If your process crashes mid-optimization, you lose all trial history otherwise.
- **Check `ax_client.get_trials_as_df()`** — Always inspect what Ax has seen. Debugging optimization starts here.

### ❌ DON'T

- **Don't optimize > 20 continuous parameters with BO** — GP surrogate degrades rapidly in high dimensions. Use parameter reduction (PCA, sensitivity analysis) first, or switch to evolutionary methods.
- **Don't ignore the Sobol phase** — Don't try to pre-fill trials with your own random points and skip Sobol. Ax's generation strategy expects to manage this internally.
- **Don't report failed trials as metric = 0** — Use `ax_client.fail_trial(trial_index)` for failures. Reporting 0 teaches the surrogate that those parameters are bad, which may be wrong.
- **Don't mix minimize=True and maximize=True expectations** — Double-check your `minimize` flag. Ax will happily optimize in the wrong direction.
- **Don't expect convergence in < 10 BO trials** — The first 5–10 BO suggestions are still exploratory. Convergence typically needs 15–30 BO trials beyond Sobol.
- **Don't use fixed parameters as optimization targets** — Fixed parameters are constants. They're in the experiment for metadata / reproducibility, not for search.

## Anti-Patterns (NEVER)

```python
from ax.service.ax_client import AxClient
import numpy as np

# ❌ BAD: Reporting failures as metric = 0 — teaches surrogate wrong signal
ax_client = AxClient()
ax_client.create_experiment(
    name="bad_example",
    parameters=[{"name": "lr", "type": "range", "bounds": [1e-5, 1.0], "log_scale": True}],
    objective_name="loss",
    minimize=True
)

for i in range(30):
    params, idx = ax_client.get_next_trial()
    try:
        loss = train_model(params["lr"])
        ax_client.complete_trial(trial_index=idx, raw_data={"loss": loss})
    except Exception:
        ax_client.complete_trial(trial_index=idx, raw_data={"loss": 0})  # ← WRONG: 0 looks great!

# ✅ GOOD: Mark failures explicitly
for i in range(30):
    params, idx = ax_client.get_next_trial()
    try:
        loss = train_model(params["lr"])
        ax_client.complete_trial(trial_index=idx, raw_data={"loss": loss})
    except Exception:
        ax_client.fail_trial(trial_index=idx)   # ← Correct: Ax ignores this trial

# ─────────────────────────────────────────────────

# ❌ BAD: Continuous range for categorical choices
ax_client.create_experiment(
    name="bad_params",
    parameters=[
        {"name": "optimizer", "type": "range", "bounds": [0, 2]},  # 0=adam, 1=sgd, 2=adamw ← WRONG
    ],
    objective_name="loss", minimize=True
)
# BO will suggest optimizer=0.73 — meaningless!

# ✅ GOOD: Choice parameter for discrete options
ax_client.create_experiment(
    name="good_params",
    parameters=[
        {"name": "optimizer", "type": "choice", "values": ["adam", "sgd", "adamw"]},
    ],
    objective_name="loss", minimize=True
)

# ─────────────────────────────────────────────────

# ❌ BAD: No log scale for orders-of-magnitude range
ax_client.create_experiment(
    name="no_log",
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-5, 1.0]},  # Linear scale!
    ],
    objective_name="loss", minimize=True
)
# BO samples: 0.3, 0.7, 0.5, 0.8 ... almost never touches 1e-5 to 1e-3 region
# where the optimum usually lives for learning rates.

# ✅ GOOD: Log scale
ax_client.create_experiment(
    name="with_log",
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-5, 1.0], "log_scale": True},
    ],
    objective_name="loss", minimize=True
)
# Now BO samples uniformly in log space: 1e-5, 3e-4, 1e-2, 7e-1 — covers the full range
```

## Parameter Search Space

```python
from ax.service.ax_client import AxClient

# ─── RANGE: Continuous or integer interval ───
parameters = [
    # Continuous (default)
    {"name": "learning_rate",   "type": "range", "bounds": [1e-5, 1.0],   "log_scale": True},
    {"name": "weight_decay",    "type": "range", "bounds": [0.0, 0.1]},
    {"name": "momentum",        "type": "range", "bounds": [0.5, 0.99]},

    # Integer range
    {"name": "num_layers",      "type": "range", "bounds": [1, 8],        "dtype": "int"},
    {"name": "hidden_dim",      "type": "range", "bounds": [32, 512],     "dtype": "int", "log_scale": True},

    # ─── CHOICE: Discrete set of options ───
    {"name": "optimizer",       "type": "choice", "values": ["adam", "sgd", "adamw", "rmsprop"]},
    {"name": "activation",      "type": "choice", "values": ["relu", "gelu", "silu", "tanh"]},
    {"name": "scheduler",       "type": "choice", "values": ["cosine", "step", "plateau"]},

    # ─── FIXED: Constant — not optimized, stored as metadata ───
    {"name": "dataset",         "type": "fixed",  "value": "CIFAR10"},
    {"name": "seed",            "type": "fixed",  "value": 42},
]

# ─── Parameter constraints: linear relations between parameters ───
# "x1 + x2 <= 1.0" — the sum of two params cannot exceed 1
# Useful for: budget allocation, mixture compositions, time budgets
ax_client = AxClient()
ax_client.create_experiment(
    name="constrained_params",
    parameters=[
        {"name": "train_frac",  "type": "range", "bounds": [0.1, 0.9]},
        {"name": "val_frac",    "type": "range", "bounds": [0.05, 0.5]},
    ],
    objective_name="accuracy",
    minimize=False,
    parameter_constraints=["train_frac + val_frac <= 1.0"]   # test_frac = 1 - train - val >= 0
)
```

## Single-Objective Optimization — Full Pipeline

```python
from ax.service.ax_client import AxClient
import numpy as np
import pandas as pd

def optimize_single_objective(
    objective_func,           # Callable: parameters_dict → float
    parameters: list[dict],   # Search space definition
    objective_name: str = "metric",
    minimize: bool = True,
    n_trials: int = 30,
    parameter_constraints: list[str] = None,
    experiment_name: str = "optimization"
) -> dict:
    """
    Full single-objective Bayesian optimization pipeline.
    Returns optimization report with best parameters, convergence history, all trials.
    """
    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        name=experiment_name,
        parameters=parameters,
        objective_name=objective_name,
        minimize=minimize,
        parameter_constraints=parameter_constraints or []
    )

    history = []

    for i in range(n_trials):
        # ASK: get next suggested parameters
        params, trial_index = ax_client.get_next_trial()

        # RUN: evaluate the objective
        try:
            value = objective_func(params)
            ax_client.complete_trial(trial_index=trial_index, raw_data={objective_name: value})
            status = "completed"
        except Exception as e:
            ax_client.fail_trial(trial_index=trial_index)
            value = None
            status = f"failed: {e}"

        history.append({
            "trial":     trial_index,
            "params":    params,
            "value":     value,
            "status":    status
        })

        # Progress
        if (i + 1) % 5 == 0:
            best_params, (best_mean, _) = ax_client.get_best_parameters()
            direction = "↓" if minimize else "↑"
            print(f"  Trial {i+1:3d}/{n_trials} | Best {objective_name}: {best_mean:.4f} {direction}")

    # Final report
    best_params, (best_mean, best_sem) = ax_client.get_best_parameters()

    report = {
        "best_parameters":  best_params,
        "best_mean":        best_mean,
        "best_sem":         best_sem,
        "history":          history,
        "ax_client":        ax_client,
        "n_completed":      sum(1 for h in history if h["status"] == "completed"),
        "n_failed":         sum(1 for h in history if "failed" in h["status"]),
    }

    print(f"\n{'='*50}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'='*50}")
    print(f"  Best {objective_name}:  {best_mean:.4f} ± {best_sem:.4f}")
    print(f"  Best parameters:")
    for k, v in best_params.items():
        print(f"    {k:25s} = {v}")
    print(f"  Completed: {report['n_completed']}, Failed: {report['n_failed']}")

    return report

# ─── Example: Rosenbrock function ───
def rosenbrock(params):
    x, y = params["x"], params["y"]
    return (1 - x)**2 + 100 * (y - x**2)**2   # Minimum at (1, 1) = 0

report = optimize_single_objective(
    objective_func=rosenbrock,
    parameters=[
        {"name": "x", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "y", "type": "range", "bounds": [-5.0, 5.0]},
    ],
    objective_name="rosenbrock",
    minimize=True,
    n_trials=30,
    experiment_name="rosenbrock_opt"
)
```

## Multi-Objective Optimization — Pareto Frontier

```python
from ax.service.ax_client import AxClient, ObjectiveProperties
import numpy as np

def optimize_multi_objective(
    objective_func,           # Callable: params → dict of {metric_name: value}
    parameters: list[dict],
    objectives: dict,         # {metric_name: ObjectiveProperties(minimize=bool)}
    n_trials: int = 40,
    experiment_name: str = "multi_obj"
) -> dict:
    """
    Multi-objective Bayesian optimization.
    Returns Pareto-optimal set: solutions where no objective can improve
    without worsening another.
    """
    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        name=experiment_name,
        parameters=parameters,
        objectives=objectives   # Multi-objective: pass dict of ObjectiveProperties
    )

    all_results = []

    for i in range(n_trials):
        params, trial_index = ax_client.get_next_trial()

        try:
            metrics = objective_func(params)   # Returns ALL metric values
            ax_client.complete_trial(trial_index=trial_index, raw_data=metrics)
            all_results.append({"trial": trial_index, "params": params, **metrics})
        except Exception:
            ax_client.fail_trial(trial_index=trial_index)

        if (i + 1) % 10 == 0:
            print(f"  Trial {i+1}/{n_trials} completed")

    # Extract Pareto frontier from trial history
    # (Ax tracks this internally — here we reconstruct for analysis)
    metric_names = list(objectives.keys())
    minimize_flags = {m: objectives[m].minimize for m in metric_names}

    df = ax_client.get_trials_as_df()

    return {
        "all_trials":    df,
        "ax_client":     ax_client,
        "objectives":    objectives,
        "metric_names":  metric_names,
    }

# ─── Example: Speed vs Accuracy tradeoff ───
def speed_accuracy_tradeoff(params):
    """Simulate: more layers = better accuracy but slower inference."""
    layers     = params["num_layers"]
    hidden     = params["hidden_dim"]
    dropout    = params["dropout"]

    # Simulated accuracy (higher layers + hidden = better, dropout helps regularize)
    accuracy = 1.0 - 0.3 * np.exp(-0.1 * layers * hidden / 100) + 0.05 * dropout
    accuracy = min(accuracy, 0.99) + 0.01 * np.random.randn()

    # Simulated latency in ms (more layers + hidden = slower)
    latency = 5.0 + 2.0 * layers + 0.05 * hidden + np.random.randn()

    return {"accuracy": accuracy, "latency": latency}

report = optimize_multi_objective(
    objective_func=speed_accuracy_tradeoff,
    parameters=[
        {"name": "num_layers",  "type": "range",  "bounds": [1, 12],    "dtype": "int"},
        {"name": "hidden_dim",  "type": "range",  "bounds": [32, 512],  "dtype": "int", "log_scale": True},
        {"name": "dropout",     "type": "range",  "bounds": [0.0, 0.5]},
    ],
    objectives={
        "accuracy": ObjectiveProperties(minimize=False),   # Maximize accuracy
        "latency":  ObjectiveProperties(minimize=True),    # Minimize latency
    },
    n_trials=40,
    experiment_name="speed_accuracy_tradeoff"
)
```

## Outcome Constraints

```python
from ax.service.ax_client import AxClient
import numpy as np

# Outcome constraints: restrict the search to regions where a METRIC
# (not a parameter) satisfies a condition.
# Example: "optimize accuracy, but only consider solutions where memory < 2GB"

ax_client = AxClient(verbose_logging=False)
ax_client.create_experiment(
    name="constrained_optimization",
    parameters=[
        {"name": "batch_size",   "type": "range", "bounds": [16, 256],    "dtype": "int", "log_scale": True},
        {"name": "num_layers",   "type": "range", "bounds": [1, 12],      "dtype": "int"},
        {"name": "hidden_dim",   "type": "range", "bounds": [64, 1024],   "dtype": "int", "log_scale": True},
    ],
    objective_name="accuracy",
    minimize=False,
    # Outcome constraint: memory usage must stay under 4096 MB
    # Format: "metric_name op value"  where op is <= or >=
    outcome_constraints=["memory_mb <= 4096.0"]
)

def train_and_measure(params):
    """Returns accuracy AND memory usage."""
    # Simulated
    accuracy   = 0.7 + 0.001 * params["num_layers"] * params["hidden_dim"] / 100
    memory_mb  = params["batch_size"] * params["hidden_dim"] * params["num_layers"] * 0.01
    return {"accuracy": accuracy, "memory_mb": memory_mb}

for i in range(30):
    params, trial_index = ax_client.get_next_trial()
    metrics = train_and_measure(params)

    # Report ALL metrics — both objective AND constraint metrics
    ax_client.complete_trial(trial_index=trial_index, raw_data=metrics)

# Best parameters that satisfy the constraint
best_params, (best_mean, best_sem) = ax_client.get_best_parameters()
print(f"Best accuracy (memory ≤ 4096 MB): {best_mean:.4f}")
print(f"Best params: {best_params}")

# Inspect all trials with constraint satisfaction
df = ax_client.get_trials_as_df()
print(f"\nConstraint satisfied: {(df['memory_mb'] <= 4096).sum()}/{len(df)} trials")
```

## Noisy Evaluations

```python
from ax.service.ax_client import AxClient
import numpy as np

# Scientific experiments are noisy. Multiple measurements of the same
# condition give different results. Ax handles this via SEM (Standard Error of Mean).
# Passing SEM tells the surrogate: "this observation has uncertainty."
# Without SEM, BO may overfit to noise.

ax_client = AxClient(verbose_logging=False)
ax_client.create_experiment(
    name="noisy_experiment",
    parameters=[
        {"name": "temperature",   "type": "range", "bounds": [200.0, 400.0]},
        {"name": "pressure",      "type": "range", "bounds": [1.0, 10.0]},
        {"name": "catalyst_conc", "type": "range", "bounds": [0.01, 1.0], "log_scale": True},
    ],
    objective_name="yield",
    minimize=False
)

def run_experiment(params, n_repeats=3):
    """
    Run experiment n_repeats times, return mean and SEM.
    In real code: this might be 3 replicate lab runs.
    """
    T  = params["temperature"]
    P  = params["pressure"]
    C  = params["catalyst_conc"]

    # Simulated noisy yield
    true_yield = 0.5 + 0.001 * T + 0.05 * P + 0.3 * np.log10(C)
    measurements = [true_yield + 0.05 * np.random.randn() for _ in range(n_repeats)]

    mean = np.mean(measurements)
    sem  = np.std(measurements, ddof=1) / np.sqrt(n_repeats)

    return mean, sem

for i in range(25):
    params, trial_index = ax_client.get_next_trial()

    mean, sem = run_experiment(params)

    # ─── KEY: Pass (mean, sem) tuple for noisy observations ───
    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data={"yield": (mean, sem)}   # Tuple: (mean, standard_error_of_mean)
    )

    if (i + 1) % 5 == 0:
        best_params, (best_mean, best_sem) = ax_client.get_best_parameters()
        print(f"  Trial {i+1:2d} | Best yield: {best_mean:.3f} ± {best_sem:.3f}")

best_params, (best_mean, best_sem) = ax_client.get_best_parameters()
print(f"\nOptimal conditions:")
for k, v in best_params.items():
    print(f"  {k:20s} = {v:.4f}")
print(f"  Predicted yield: {best_mean:.3f} ± {best_sem:.3f}")
```

## Practical Workflows

### 1. ML Hyperparameter Tuning

```python
from ax.service.ax_client import AxClient
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def full_hpo_pipeline(X_train, y_train, X_val, y_val, n_trials=35):
    """
    Bayesian HPO for a PyTorch MLP.
    Search space: lr, weight_decay, hidden_dim, num_layers, dropout, optimizer.
    """
    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        name="mlp_hpo",
        parameters=[
            {"name": "lr",            "type": "range",   "bounds": [1e-5, 1e-1],  "log_scale": True},
            {"name": "weight_decay",  "type": "range",   "bounds": [1e-6, 1e-2],  "log_scale": True},
            {"name": "hidden_dim",    "type": "range",   "bounds": [32, 512],     "dtype": "int", "log_scale": True},
            {"name": "num_layers",    "type": "range",   "bounds": [1, 5],        "dtype": "int"},
            {"name": "dropout",       "type": "range",   "bounds": [0.0, 0.5]},
            {"name": "optimizer",     "type": "choice",  "values": ["adam", "adamw", "sgd"]},
        ],
        objective_name="val_accuracy",
        minimize=False
    )

    best_acc = 0
    best_config = None

    for trial_num in range(n_trials):
        params, trial_index = ax_client.get_next_trial()

        try:
            # Build model
            layers = []
            in_dim = X_train.shape[1]
            for _ in range(params["num_layers"]):
                layers.extend([
                    nn.Linear(in_dim, params["hidden_dim"]),
                    nn.ReLU(),
                    nn.Dropout(params["dropout"])
                ])
                in_dim = params["hidden_dim"]
            layers.append(nn.Linear(in_dim, len(np.unique(y_train))))
            model = nn.Sequential(*layers)

            # Optimizer
            optim_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
                         "sgd": torch.optim.SGD}[params["optimizer"]]
            optimizer = optim_cls(model.parameters(), lr=params["lr"],
                                  weight_decay=params["weight_decay"])

            # Train (simplified — 20 epochs)
            criterion = nn.CrossEntropyLoss()
            X_t = torch.FloatTensor(X_train)
            y_t = torch.LongTensor(y_train)
            dataset = TensorDataset(X_t, y_t)
            loader  = DataLoader(dataset, batch_size=64, shuffle=True)

            model.train()
            for epoch in range(20):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

            # Validate
            model.eval()
            with torch.no_grad():
                val_preds = model(torch.FloatTensor(X_val)).argmax(dim=1).numpy()
            val_acc = accuracy_score(y_val, val_preds)

            ax_client.complete_trial(trial_index=trial_index,
                                     raw_data={"val_accuracy": val_acc})

            if val_acc > best_acc:
                best_acc   = val_acc
                best_config = params

        except Exception as e:
            ax_client.fail_trial(trial_index=trial_index)
            print(f"  Trial {trial_num} failed: {e}")
            continue

        if (trial_num + 1) % 5 == 0:
            print(f"  Trial {trial_num+1:2d}/{n_trials} | Best val_acc: {best_acc:.4f}")

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print(f"Best config: {best_config}")
    return ax_client, best_config

# ─── Usage ───
# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=10, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# ax_client, best = full_hpo_pipeline(X_train, y_train, X_val, y_val)
```

### 2. Chemical Reaction Optimization

```python
from ax.service.ax_client import AxClient
import numpy as np

def optimize_reaction(n_trials=40):
    """
    Optimize a chemical reaction: maximize yield subject to purity constraint.
    Parameters: temperature, catalyst loading, solvent ratio, reaction time.
    Metrics: yield (maximize), purity (must be >= 95%)
    """
    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        name="reaction_optimization",
        parameters=[
            {"name": "temperature",     "type": "range", "bounds": [40.0, 120.0]},
            {"name": "catalyst_loading","type": "range", "bounds": [0.01, 0.5], "log_scale": True},
            {"name": "solvent_ratio",   "type": "range", "bounds": [1.0, 10.0]},
            {"name": "reaction_time",   "type": "range", "bounds": [0.5, 24.0], "log_scale": True},
            {"name": "base",            "type": "choice","values": ["K2CO3", "NaOH", "Et3N", "DIPEA"]},
        ],
        objective_name="yield",
        minimize=False,
        outcome_constraints=["purity >= 95.0"]   # Purity must be ≥ 95%
    )

    def simulate_reaction(params):
        """Simulated reaction with realistic noise and constraints."""
        T  = params["temperature"]
        C  = params["catalyst_loading"]
        S  = params["solvent_ratio"]
        t  = params["reaction_time"]
        base_eff = {"K2CO3": 0.8, "NaOH": 1.0, "Et3N": 0.6, "DIPEA": 0.9}[params["base"]]

        # Simulated yield: peaks at optimal conditions
        yield_raw = (
            0.3
            + 0.3 * np.exp(-((T - 85)**2) / 200)      # Temperature optimum ~85°C
            + 0.2 * np.log10(C) / np.log10(0.5)        # Higher catalyst = more yield
            + 0.1 * base_eff
            + 0.05 * np.log10(t)                        # Longer time = more conversion
            - 0.05 * (S - 3)**2 / 10                   # Solvent optimum ~3
        )
        yield_pct  = np.clip(yield_raw * 100, 5, 98) + 2.0 * np.random.randn()

        # Purity: high temperature hurts purity (side reactions)
        purity = 99.0 - 0.05 * T - 0.01 * (t**1.5) + 1.0 * np.random.randn()
        purity = np.clip(purity, 60, 99.5)

        return {"yield": yield_pct, "purity": purity}

    for i in range(n_trials):
        params, trial_index = ax_client.get_next_trial()
        metrics = simulate_reaction(params)
        ax_client.complete_trial(trial_index=trial_index, raw_data=metrics)

        if (i + 1) % 10 == 0:
            best_params, (best_mean, _) = ax_client.get_best_parameters()
            print(f"  Trial {i+1:2d}/{n_trials} | Best yield (purity≥95%): {best_mean:.1f}%")

    # Report
    best_params, (best_mean, best_sem) = ax_client.get_best_parameters()
    df = ax_client.get_trials_as_df()
    feasible = df[df["purity"] >= 95.0]

    print(f"\n{'='*55}")
    print(f"  REACTION OPTIMIZATION COMPLETE")
    print(f"{'='*55}")
    print(f"  Best yield (purity ≥ 95%):  {best_mean:.1f}% ± {best_sem:.1f}%")
    print(f"  Feasible trials:            {len(feasible)}/{len(df)}")
    print(f"  Optimal conditions:")
    for k, v in best_params.items():
        print(f"    {k:22s} = {v}")

    return ax_client

# ax_client = optimize_reaction()
```

### 3. Multi-Objective Material Design

```python
from ax.service.ax_client import AxClient, ObjectiveProperties
import numpy as np

def material_design(n_trials=50):
    """
    Multi-objective optimization for material composition.
    Maximize: strength AND thermal conductivity
    Constraint: density ≤ 3.0 g/cm³ (lightweight requirement)

    This demonstrates the Pareto frontier: you can't maximize both
    strength and conductivity simultaneously — Ax finds the tradeoff curve.
    """
    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        name="material_design",
        parameters=[
            {"name": "al_fraction",  "type": "range", "bounds": [0.0, 1.0]},   # Aluminum
            {"name": "ti_fraction",  "type": "range", "bounds": [0.0, 1.0]},   # Titanium
            {"name": "c_fraction",   "type": "range", "bounds": [0.0, 0.3]},   # Carbon fiber
            {"name": "sintering_T",  "type": "range", "bounds": [800.0, 1500.0]},
        ],
        objectives={
            "strength":           ObjectiveProperties(minimize=False),
            "thermal_conductivity": ObjectiveProperties(minimize=False),
        },
        # Composition must sum to ≤ 1 (remaining = matrix)
        parameter_constraints=["al_fraction + ti_fraction + c_fraction <= 1.0"],
        outcome_constraints=["density <= 3.0"]
    )

    def simulate_material(params):
        al  = params["al_fraction"]
        ti  = params["ti_fraction"]
        c   = params["c_fraction"]
        T   = params["sintering_T"]

        # Simulated properties (with noise)
        strength = (
            100 * al + 200 * ti + 150 * c
            + 0.05 * T * (al + ti)
            + 10 * np.random.randn()
        )
        thermal_cond = (
            200 * al + 20 * ti + 5 * c       # Al conducts well, Ti poorly
            - 0.02 * T * ti                   # High T degrades Ti conductivity
            + 3 * np.random.randn()
        )
        density = 2.7 * al + 4.5 * ti + 1.8 * c + 0.1 * np.random.randn()

        return {
            "strength":            max(strength, 0),
            "thermal_conductivity": max(thermal_cond, 0),
            "density":             density
        }

    for i in range(n_trials):
        params, trial_index = ax_client.get_next_trial()
        metrics = simulate_material(params)
        ax_client.complete_trial(trial_index=trial_index, raw_data=metrics)

    # Analyze Pareto frontier
    df = ax_client.get_trials_as_df()
    feasible = df[df["density"] <= 3.0].copy()

    # Simple Pareto front extraction
    pareto = []
    for _, row in feasible.iterrows():
        dominated = False
        for _, other in feasible.iterrows():
            if (other["strength"] >= row["strength"] and
                other["thermal_conductivity"] >= row["thermal_conductivity"] and
                (other["strength"] > row["strength"] or
                 other["thermal_conductivity"] > row["thermal_conductivity"])):
                dominated = True
                break
        if not dominated:
            pareto.append(row)

    pareto_df = type(df)(pareto)

    print(f"\n{'='*55}")
    print(f"  MATERIAL DESIGN — PARETO FRONTIER")
    print(f"{'='*55}")
    print(f"  Total trials:    {len(df)}")
    print(f"  Feasible:        {len(feasible)} (density ≤ 3.0)")
    print(f"  Pareto-optimal:  {len(pareto_df)} solutions")
    print(f"\n  Pareto front (strength ↑, thermal conductivity ↑):")
    pareto_sorted = pareto_df.sort_values("strength", ascending=False)
    for _, row in pareto_sorted.iterrows():
        print(f"    strength={row['strength']:7.1f} | "
              f"thermal_cond={row['thermal_conductivity']:6.1f} | "
              f"density={row['density']:.2f}")

    return ax_client, pareto_df

# ax_client, pareto = material_design()
```

### 4. Cost-Aware Experiment Design

```python
from ax.service.ax_client import AxClient
import numpy as np

def cost_aware_optimization(budget=500.0, n_trials=30):
    """
    Each experiment has a COST. Total budget is limited.
    Use outcome constraints to stay within budget.
    Ax learns which parameter regions are expensive and avoids them
    unless they're likely to yield high returns.
    """
    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        name="cost_aware",
        parameters=[
            {"name": "num_samples",   "type": "range", "bounds": [100, 10000],  "dtype": "int", "log_scale": True},
            {"name": "resolution",    "type": "range", "bounds": [0.1, 2.0]},
            {"name": "method",        "type": "choice","values": ["monte_carlo", "quasi_monte_carlo", "numerical"]},
        ],
        objective_name="accuracy",
        minimize=False,
        outcome_constraints=[f"cumulative_cost <= {budget}"]
    )

    cumulative_cost = 0.0

    # Cost model: more samples + higher resolution = more expensive
    cost_model = {
        "monte_carlo":       lambda n, r: 0.01 * n * r,
        "quasi_monte_carlo": lambda n, r: 0.015 * n * r,
        "numerical":         lambda n, r: 0.05 * n * r**2,
    }

    for i in range(n_trials):
        params, trial_index = ax_client.get_next_trial()

        # Compute cost BEFORE running (if we can estimate it)
        cost_fn  = cost_model[params["method"]]
        trial_cost = cost_fn(params["num_samples"], params["resolution"])

        if cumulative_cost + trial_cost > budget:
            # Over budget — fail the trial
            ax_client.fail_trial(trial_index=trial_index)
            print(f"  Trial {i+1}: SKIPPED (cost {trial_cost:.1f} would exceed budget)")
            continue

        # Simulated accuracy
        accuracy = (
            0.5
            + 0.3 * np.log10(params["num_samples"]) / 4
            + 0.15 * params["resolution"]
            + {"monte_carlo": 0.0, "quasi_monte_carlo": 0.05, "numerical": 0.1}[params["method"]]
            + 0.02 * np.random.randn()
        )
        accuracy = min(accuracy, 0.999)
        cumulative_cost += trial_cost

        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data={
                "accuracy":        accuracy,
                "cumulative_cost": cumulative_cost
            }
        )

        if (i + 1) % 5 == 0:
            print(f"  Trial {i+1:2d} | Budget used: {cumulative_cost:.1f}/{budget} | "
                  f"Best accuracy so far: {max(r['accuracy'] for r in [{'accuracy': accuracy}]):.4f}")

    best_params, (best_mean, _) = ax_client.get_best_parameters()
    print(f"\nBest accuracy: {best_mean:.4f} | Total cost: {cumulative_cost:.1f}/{budget}")
    print(f"Best params: {best_params}")
    return ax_client

# ax_client = cost_aware_optimization(budget=500.0)
```

## Visualization and Analysis

```python
from ax.service.ax_client import AxClient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_optimization_history(ax_client: AxClient, objective_name: str,
                              minimize: bool = True, title: str = None):
    """Plot convergence: best metric found so far vs trial number."""
    df = ax_client.get_trials_as_df()
    df = df[df["status"] == "COMPLETED"].sort_values("trial_index")

    values = df[objective_name].values
    # Running best
    if minimize:
        running_best = np.minimum.accumulate(values)
    else:
        running_best = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["trial_index"], values,         color='steelblue', alpha=0.5, label='All trials', zorder=2)
    ax.plot(   df["trial_index"], running_best,   color='darkred',   linewidth=2, label='Running best', zorder=3)

    # Mark the Sobol/BO boundary (approximate: first ~5-10 trials are Sobol)
    n_sobol = max(5, 2 * (len(df.columns) - 3))   # Heuristic
    ax.axvline(x=n_sobol, color='gray', linestyle='--', alpha=0.5)
    ax.text(n_sobol + 0.3, ax.get_ylim()[0], 'Sobol → BO', fontsize=9, color='gray')

    direction = "↓ minimize" if minimize else "↑ maximize"
    ax.set_xlabel("Trial index")
    ax.set_ylabel(objective_name)
    ax.set_title(title or f"Optimization convergence ({direction})")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_pareto_frontier(ax_client: AxClient,
                         metric_x: str, metric_y: str,
                         constraint_metric: str = None, constraint_value: float = None):
    """Plot Pareto frontier for two objectives."""
    df = ax_client.get_trials_as_df()
    df = df[df["status"] == "COMPLETED"]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Filter feasible if constraint given
    if constraint_metric and constraint_value:
        feasible   = df[df[constraint_metric] <= constraint_value]
        infeasible = df[df[constraint_metric] > constraint_value]
        ax.scatter(infeasible[metric_x], infeasible[metric_y],
                   color='lightgray', alpha=0.5, s=40, label='Infeasible', zorder=1)
    else:
        feasible = df

    # Extract Pareto front (both maximized for simplicity — negate if minimizing)
    pareto_mask = np.ones(len(feasible), dtype=bool)
    for i, (_, row_i) in enumerate(feasible.iterrows()):
        for j, (_, row_j) in enumerate(feasible.iterrows()):
            if i != j:
                if (row_j[metric_x] >= row_i[metric_x] and
                    row_j[metric_y] >= row_i[metric_y] and
                    (row_j[metric_x] > row_i[metric_x] or row_j[metric_y] > row_i[metric_y])):
                    pareto_mask[i] = False
                    break

    pareto    = feasible[pareto_mask]
    non_pareto = feasible[~pareto_mask]

    ax.scatter(non_pareto[metric_x], non_pareto[metric_y],
               color='steelblue', alpha=0.4, s=50, label='Feasible (non-Pareto)', zorder=2)
    ax.scatter(pareto[metric_x], pareto[metric_y],
               color='darkred', s=100, edgecolors='black', linewidths=1.5,
               label=f'Pareto optimal ({len(pareto)})', zorder=3)

    # Connect Pareto front with a line (sorted by x)
    pareto_sorted = pareto.sort_values(metric_x)
    ax.plot(pareto_sorted[metric_x], pareto_sorted[metric_y],
            'r--', alpha=0.6, linewidth=1.5)

    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    ax.set_title("Pareto Frontier")
    ax.legend()
    plt.tight_layout()
    plt.show()

# ─── Usage ───
# plot_optimization_history(ax_client, "loss", minimize=True)
# plot_pareto_frontier(ax_client, "strength", "thermal_conductivity",
#                      constraint_metric="density", constraint_value=3.0)
```

## Common Pitfalls and Solutions

### Optimization Goes in Wrong Direction

```python
# ❌ PROBLEM: minimize=True but you want to maximize accuracy
ax_client.create_experiment(
    name="wrong_direction",
    parameters=[...],
    objective_name="accuracy",
    minimize=True    # ← WRONG for accuracy! Ax will try to get accuracy → 0
)

# ✅ SOLUTION: Double-check minimize flag matches your goal
# maximize accuracy → minimize=False
# minimize loss    → minimize=True
ax_client.create_experiment(
    name="correct_direction",
    parameters=[...],
    objective_name="accuracy",
    minimize=False   # ← Correct: maximize accuracy
)
```

### BO Suggests Only Extreme Values

```python
# ❌ SYMPTOM: After Sobol, BO only suggests values at bounds
# (e.g., lr=1e-5 or lr=1.0, never in between)
# CAUSE: Missing log_scale on an order-of-magnitude parameter.
# The GP sees the [1e-5, 1.0] range as almost entirely "near 1.0"
# in linear space. It has no resolution at the low end.

# ✅ FIX: Add log_scale=True
{"name": "lr", "type": "range", "bounds": [1e-5, 1.0], "log_scale": True}
# Now the GP sees the range as uniform in log space.
```

### Too Few Trials — "Optimization Hasn't Converged"

```python
# ❌ SYMPTOM: 15 trials, best hasn't improved in 5 trials.
# CAUSE: Budget too small. Sobol uses ~5-10 trials.
# BO needs 10-20 MORE trials to find the optimum.
# Total minimum: 20-30 trials.

# ✅ RULE OF THUMB:
# n_trials >= max(30, 10 * num_continuous_parameters)
# 2 params  → 30 trials minimum
# 5 params  → 50 trials minimum
# 10 params → 100 trials minimum
```

### Constraint Metric Not Reported

```python
# ❌ PROBLEM: outcome_constraints=["memory_mb <= 4096"]
# but complete_trial only reports the objective:
ax_client.complete_trial(trial_index=idx, raw_data={"accuracy": 0.95})
# Ax doesn't know memory_mb → constraint is ignored!

# ✅ SOLUTION: Always report ALL metrics — objective AND constraint
ax_client.complete_trial(trial_index=idx, raw_data={
    "accuracy":  0.95,
    "memory_mb": 2048.0    # ← Must include constraint metric
})
```

### All Trials Fail

```python
# ❌ SYMPTOM: ax_client.fail_trial() called for every trial.
# CAUSE: Usually a bug in the objective function, not in Ax.
# Debug the objective function FIRST with a single call.

# ✅ DIAGNOSTIC: Test the objective before optimization
params_test = {p["name"]: (p["bounds"][0] if p["type"] == "range"
               else p["values"][0]) for p in parameters}
try:
    result = objective_func(params_test)
    print(f"Test call succeeded: {result}")
except Exception as e:
    print(f"Objective function is broken: {e}")   # Fix THIS first
```

---

Ax's core value is the **closed loop**: it doesn't just optimize — it learns from every trial and systematically narrows the search. The Sobol → BO transition happens automatically, the surrogate model gets better with each observation, and the acquisition function balances "exploit what looks promising" against "explore what's uncertain." Master the ask-tell pattern, use log_scale where it matters, report SEM for noisy measurements, and always include constraint metrics in `raw_data` — and you have a production-grade adaptive experimentation engine.