---
name: dask
description: A flexible library for parallel computing in Python. It scales Python libraries like NumPy, pandas, and scikit-learn to multi-core systems or distributed clusters. Features lazy evaluation and task scheduling for data that exceeds RAM capacity. Use for out-of-core computing, parallel processing, distributed computing, large-scale data analysis, dask.array, dask.dataframe, dask.delayed, dask.bag, task scheduling, lazy evaluation, and scaling beyond memory limits.
version: 2024.1
license: BSD-3-Clause
---

# Dask - Scalable Parallel Computing

Dask provides high-level collections (Arrays, DataFrames, Bags) that mimic the APIs of NumPy and pandas but operate in parallel on data sets that are larger than memory.

## When to Use

- Processing datasets that don't fit in RAM (Out-of-core computing).
- Speeding up computations by using all available CPU cores.
- Parallelizing custom Python functions or complex workflows (dask.delayed).
- Scaling machine learning pipelines to large clusters.
- Handling large-scale arrays in physics, climate science, or imaging.
- Analyzing massive log files or unstructured data (dask.bag).

## Reference Documentation

**Official docs**: https://docs.dask.org/  
**Dask Examples**: https://examples.dask.org/  
**Search patterns**: `dask.dataframe`, `dask.array`, `dask.delayed`, `client.compute`, `dask.distributed`

## Core Principles

### Lazy Evaluation

Dask doesn't compute results immediately. Instead, it builds a Task Graph. Actual computation only happens when you explicitly call `.compute()` or `.persist()`.

### Chunks and Partitions

- **Dask Array**: Composed of many small NumPy arrays called chunks.
- **Dask DataFrame**: Composed of many small pandas DataFrames called partitions.

### Use Dask For

| Collection | Analogy | Use Case |
|------------|---------|----------|
| `dask.array` | NumPy | Large-scale multidimensional math. |
| `dask.dataframe` | pandas | Large CSV/Parquet/SQL tables. |
| `dask.bag` | Lists/Toolz | Unstructured data (JSON, Logs). |
| `dask.delayed` | Functions | Custom parallel logic. |

### Do NOT Use For

- Data that fits easily in RAM (pandas/NumPy are faster due to lower overhead).
- Simple tasks where multiprocessing or concurrent.futures suffice.
- Situations where low-latency response is required (Dask adds scheduling overhead).

## Quick Reference

### Installation

```bash
pip install "dask[complete]"
```

### Standard Imports

```python
import dask.array as da
import dask.dataframe as dd
from dask import delayed, compute
from dask.distributed import Client
```

### Basic Pattern - Initializing a Local Cluster

```python
from dask.distributed import Client

# Setup local cluster and dashboard
client = Client() 
print(client.dashboard_link) # View real-time computation graph
```

## Critical Rules

### ✅ DO

- **Use the Dashboard** - Always monitor the Dask dashboard to find bottlenecks (red blocks = bad).
- **Chunk thoughtfully** - Aim for chunk sizes of 100MB to 250MB. Too small = high overhead; too large = memory errors.
- **Prefer Parquet** - Use Parquet instead of CSV for DataFrames; it supports efficient metadata and partitioning.
- **Call `.persist()` on reused data** - If you use the same intermediate result multiple times, persist it in memory.
- **Let Dask handle the graph** - Avoid calling `.compute()` too early; try to keep calculations lazy as long as possible.
- **Use `map_partitions`** - For custom logic on DataFrames, use this to apply pandas functions directly to each chunk.

### ❌ DON'T

- **Compute too often** - Every `.compute()` triggers the entire graph execution and pulls data into RAM.
- **Send large data to workers** - Use `client.scatter` for large objects needed by all workers instead of passing them as arguments.
- **Iterate over rows** - `for row in dask_df` is incredibly slow; use vectorized operations.
- **Use Dask if pandas is enough** - Dask is slower for small data due to scheduling time.

## Anti-Patterns (NEVER)

```python
import dask.dataframe as dd

# ❌ BAD: Computing a large result into a local variable
# This will crash your local machine by filling RAM
result = dd_df.compute() 

# ✅ GOOD: Compute only what you need (aggregations)
mean_val = dd_df['column'].mean().compute()

# ❌ BAD: Too many small tasks (Task Overhead)
# result = [delayed(inc)(i) for i in range(1000000)] # 1 million tasks is too much

# ✅ GOOD: Batch tasks together or use Dask Collections
import dask.array as da
x = da.arange(1000000, chunks=10000) # Only 100 tasks

# ❌ BAD: Hardcoding workers' file paths
# dd.read_csv('/Users/me/data.csv') # Workers on other machines can't see this path!

# ✅ GOOD: Use shared storage (S3, HDFS, NFS)
# dd.read_csv('s3://my-bucket/data.csv')
```

## Dask Array (dask.array)

### Scaling NumPy

```python
import dask.array as da

# Create a large random array (100GB)
x = da.random.random((100000, 100000), chunks=(10000, 10000))

# Perform operations (Lazy)
y = x + x.T
z = y[::2, :5000].mean(axis=0)

# Compute result
result = z.compute()
```

## Dask DataFrame (dask.dataframe)

### Scaling pandas

```python
import dask.dataframe as dd

# Load massive dataset
df = dd.read_csv('data/*.csv')

# Filtering and Grouping
result = (df[df['value'] > 0]
          .groupby('category')
          .amount.sum())

# Execute
final_amounts = result.compute()

# Convert from pandas to dask
import pandas as pd
pdf = pd.DataFrame(...)
ddf = dd.from_pandas(pdf, npartitions=10)
```

## Dask Delayed (dask.delayed)

### Parallelizing Custom Code

```python
from dask import delayed

@delayed
def load(filename):
    ...
    return data

@delayed
def process(data):
    ...
    return result

@delayed
def summarize(results):
    return sum(results)

# Build graph
filenames = ['file1.csv', 'file2.csv', 'file3.csv']
outputs = [process(load(f)) for f in filenames]
total = summarize(outputs)

# Visualize graph (requires graphviz)
# total.visualize()

# Execute in parallel
final_sum = total.compute()
```

## Machine Learning with Dask (dask-ml)

```python
from dask_ml.preprocessing import StandardScaler
from dask_ml.linear_model import LogisticRegression
from dask_ml.model_selection import train_test_split

# Scaling to large data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dask)

# Training on large datasets (Parallel SGD)
model = LogisticRegression()
model.fit(X_dask, y_dask)

# Scikit-learn wrapper (for small data, parallelizing search)
from sklearn.ensemble import RandomForestClassifier
from dask_ml.model_selection import GridSearchCV

clf = RandomForestClassifier()
grid = GridSearchCV(clf, param_grid, cv=3)
grid.fit(X, y) # Grid search runs in parallel across cluster
```

## Practical Workflows

### 1. Massive Log Processing (Bag)

```python
import dask.bag as db
import json

def analyze_logs(pattern):
    # Read unstructured text files
    b = db.read_text('logs/2023-*.log')
    
    # Parse JSON and filter
    records = b.map(json.loads).filter(lambda x: x['level'] == 'ERROR')
    
    # Extract specific field and count frequencies
    counts = records.pluck('message').frequencies()
    
    return counts.compute()
```

### 2. Large Scale Imaging (Array)

```python
def process_satellite_images(da_stack):
    """Calculate NDVI anomaly across time on 1TB of data."""
    # da_stack is a 3D dask array (time, x, y)
    
    # Simple vectorized math (Parallel)
    climatology = da_stack.mean(axis=0)
    anomaly = da_stack - climatology
    
    # Save results directly to disk without loading into RAM
    anomaly.to_zarr('anomalies.zarr')
```

### 3. Cleaning Data with Method Chaining

```python
def clean_dataset(ddf):
    return (ddf
            .dropna(subset=['id'])
            .fillna({'status': 'unknown'})
            .assign(timestamp=dd.to_datetime(ddf['time_str']))
            .groupby('user_id')
            .last()
            .persist()) # Keep in memory for fast future use
```

## Performance Optimization

### The Dask Dashboard Guide

- **Progress Bar**: Shows how many tasks are finished.
- **Task Stream**: Shows which worker is doing what. White space = idle workers (bad).
- **Memory Plot**: Shows RAM usage. If it turns orange/red, workers are hitting limits.
- **Worker Table**: Check for skewed data distribution.

### Optimizing Data Storage

- **Zarr**: Best for N-dimensional arrays.
- **Parquet**: Best for tabular DataFrames.
- **Compression**: Use snappy or lz4 for a balance between speed and size.

## Common Pitfalls and Solutions

### The "Worker Lost" Error

**Problem**: Workers crash because they ran out of RAM.

**Solution**: Decrease chunk size or use a machine with more memory. Check for data skew.

### Serialization Errors (Pickle)

**Problem**: Dask can't send your custom object to workers.

**Solution**: Use `dask.distributed.Client.register_plugin` or ensure classes are defined in a separate file accessible by workers.

### "Too Many Tasks" Warning

**Problem**: You created 1,000,000+ tiny tasks.

**Solution**: Re-chunk your data into larger pieces. Use `dask_array.rechunk()` or `dask_df.repartition()`.

## Best Practices

1. Always monitor the Dask dashboard during development to identify bottlenecks.
2. Choose chunk sizes carefully - aim for 100-250MB per chunk for optimal performance.
3. Use Parquet format for DataFrames instead of CSV for better performance and metadata support.
4. Persist intermediate results that are reused multiple times to avoid recomputation.
5. Keep computations lazy as long as possible - only call `.compute()` when you need the final result.
6. Use `map_partitions` for custom pandas operations on Dask DataFrames.
7. Avoid iterating over rows in Dask DataFrames - use vectorized operations instead.
8. Use shared storage (S3, HDFS, NFS) when working with distributed clusters.
9. Batch small tasks together to avoid task overhead.
10. Don't use Dask for data that fits in RAM - pandas/NumPy are faster for small datasets.

Dask transforms Python from a single-threaded scripting language into a world-class system for distributed computing. It is the bridge between a researcher's laptop and a high-performance compute cluster.
