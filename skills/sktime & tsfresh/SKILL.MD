---

name: sktime-tsfresh
description: Time series machine learning layer (Tier 1): integration of **sktime** and **tsfresh** for building production-grade pipelines that transform raw time series into tabular feature representations suitable for classical machine-learning models. *sktime* provides a unified, sklearn-compatible interface for time-series data types, transformations, and pipelines, while *tsfresh* enables large-scale automated extraction of statistical, spectral, and autocorrelation features, with optional statistically grounded feature relevance selection (FRESH).
Use when: converting time series into tabular features for any sklearn-compatible estimator; interpretability of features is important; many series/sensors must be processed consistently; automated feature generation is preferred over manual engineering; multiple feature sources (e.g., tsfresh + rolling/lag features) must be combined in a single leakage-safe pipeline.
Do not use as a default end-to-end forecasting solution: for pure forecasting tasks, sktime forecasters are often more appropriate. tsfresh is primarily a **feature-extraction and reduction layer**, not a forecasting model.
version: sktime 0.40.x; tsfresh 0.21.x
license: sktime BSD-3-Clause; tsfresh MIT
-----------------------------------------

# sktime + tsfresh — Time Series Feature Extraction Layer

This layer addresses a central applied problem in time-series machine learning: **how to transform raw temporal data into informative, reproducible, and leakage-free tabular features suitable for downstream learning algorithms**.

* **sktime** supplies standardized time-series data containers (Series, Panel, Hierarchical), transformation APIs, and pipeline composition compatible with the sklearn ecosystem.
* **tsfresh** provides automated extraction of a comprehensive set of descriptive features from time series, together with statistically motivated relevance filtering.
* sktime offers native wrappers (`TSFreshFeatureExtractor`, `TSFreshRelevantFeatureExtractor`) that integrate tsfresh seamlessly into sktime pipelines.

## Conceptual Model

```
TIME SERIES DATA
        ↓
FEATURE EXTRACTION (tsfresh via sktime)
        ↓
OPTIONAL FEATURE SELECTION (FRESH)
        ↓
TABULAR ML MODEL (sklearn-compatible)
```

All steps are embedded in a single pipeline so that feature extraction and selection are performed **only on training folds** during cross-validation, preventing information leakage.

## Data Representation

sktime operates with abstract *scitypes* (Series, Panel, Hierarchical) and multiple concrete representations.
For feature extraction, the most common format is a **nested pandas DataFrame (Panel)**:

* Rows correspond to instances (e.g., subjects, devices).
* Columns correspond to channels or variables.
* Each cell contains a univariate time series (e.g., `pd.Series` or `np.ndarray`).

Correct data representation is essential; many practical issues arise from mismatched scitypes rather than from modeling choices.

## Core Components

* **`TSFreshFeatureExtractor`**
  Performs automated feature extraction using tsfresh within a sktime transformer interface.

* **`TSFreshRelevantFeatureExtractor`**
  Extends extraction with relevance filtering based on the FRESH algorithm, reducing dimensionality while retaining informative features.

* **Pipeline composition**
  sktime pipelines and `FeatureUnion` enable sequential and parallel composition of multiple transformers and estimators.

## Canonical Pipeline Pattern

Below is a standard pattern for time-series classification using automated feature extraction:

```python
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sktime.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

tsfresh = TSFreshFeatureExtractor(
    default_fc_parameters="efficient",
    n_jobs=-1,
    disable_progressbar=True,
)

classifier = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
)

pipeline = make_pipeline(tsfresh, classifier)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
```

This formulation ensures that feature extraction occurs independently within each training fold.

## Feature Relevance Selection

High-dimensional feature spaces are common with tsfresh.
`TSFreshRelevantFeatureExtractor` integrates relevance testing directly into the pipeline:

```python
from sktime.transformations.panel.tsfresh import TSFreshRelevantFeatureExtractor
from sktime.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

pipeline = make_pipeline(
    TSFreshRelevantFeatureExtractor(
        default_fc_parameters="efficient",
        n_jobs=-1,
        disable_progressbar=True,
    ),
    LogisticRegression(max_iter=2000),
)
```

This approach reduces overfitting risk and improves interpretability without violating cross-validation assumptions.

## Combining Multiple Feature Sources

sktime supports parallel feature extraction through `FeatureUnion`, enabling the combination of tsfresh features with domain-specific or alternative time-series representations (e.g., lagged statistics, ROCKET-style features):

```python
from sktime.transformations.compose import FeatureUnion
from sktime.pipeline import make_pipeline

features = FeatureUnion([
    ("tsfresh", TSFreshFeatureExtractor(default_fc_parameters="efficient", n_jobs=-1)),
    # ("custom", CustomTimeSeriesTransformer()),
])

pipeline = make_pipeline(features, RandomForestClassifier())
```

The resulting feature matrix concatenates outputs from all transformers in a leakage-safe manner.

## Practical Configuration Guidelines

### Feature Sets

* **`minimal`**: fast, diagnostic baseline.
* **`efficient`**: recommended default, balancing coverage and cost.
* **`comprehensive`**: exhaustive but computationally expensive; use only with sufficient resources.

### Performance

* Use `n_jobs=-1` for parallel computation.
* Adjust `chunksize` for large datasets to reduce scheduling overhead.
* Disable progress bars in automated or production environments.

### Missing Values

* Configure imputation functions when necessary; unhandled NaNs propagate into feature values and can destabilize downstream models.

## Best Practices

**Do**

* Perform feature extraction and selection inside pipelines.
* Validate models using time-aware or stratified cross-validation as appropriate.
* Monitor the number and distribution of generated features for stability and drift.

**Avoid**

* Extracting features on the full dataset before splitting.
* Starting directly with exhaustive feature sets without baseline evaluation.
* Ignoring data representation requirements.

## Scope and Limitations

This layer is best suited for **feature-based reduction of time series** for classical machine-learning models, especially when interpretability and modularity are priorities.
It is not intended as a universal solution for all forecasting problems, particularly those requiring end-to-end temporal modeling or strict real-time constraints.

