---
name: scanpy
description: Scalable toolkit for analyzing single-cell gene expression data. Built on top of Anndata, focusing on clustering, trajectory inference, and visualization.
version: 1.9
license: BSD-3-Clause
---

# Scanpy - Single-Cell Analysis

Scanpy processes high-dimensional biological data, reducing it via PCA/UMAP to identify rare cell populations in tissues or microbiomes.

## When to Use

- Analyzing single-cell RNA sequencing (scRNA-seq) data.
- Identifying cell types and states in heterogeneous tissues.
- Reconstructing developmental trajectories.
- Comparing cell populations between conditions.
- Discovering rare cell types.

## Core Principles

### AnnData Format

Scanpy uses AnnData objects that store expression matrix, cell metadata, and gene annotations together.

### Dimensionality Reduction

High-dimensional gene expression (20,000+ genes) is reduced to 2D/3D for visualization (PCA → UMAP/t-SNE).

### Clustering

Cells are grouped by similarity in gene expression space to identify cell types.

## Quick Reference

### Standard Imports

```python
import scanpy as sc
import pandas as pd
import numpy as np
```

### Basic Patterns

```python
# 1. Load dataset (AnnData object)
adata = sc.read_h5ad("cells.h5ad")
# Or: adata = sc.read_10x_mtx("path/to/mtx")

# 2. QC and Normalization
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 3. Dimensionality Reduction & Visualization
sc.pp.highly_variable_genes(adata)
sc.tl.pca(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['cell_type', 'gene_A'])

# 4. Clustering
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color='leiden')
```

## Critical Rules

### ✅ DO

- **Set scanpy settings** - Use `sc.settings.verbosity = 3` for progress info.
- **Filter low-quality cells** - Remove cells with too few genes or high mitochondrial content.
- **Normalize before analysis** - Account for sequencing depth differences.
- **Use highly variable genes** - Focus analysis on informative genes.

### ❌ DON'T

- **Don't skip QC** - Low-quality cells can dominate clustering.
- **Don't use raw counts for PCA** - Always normalize and log-transform first.
- **Don't ignore batch effects** - Use batch correction (e.g., `sc.pp.harmony_integrate`) when combining datasets.

## Advanced Patterns

### Trajectory Inference

```python
import cellrank as cr

# Reconstruct developmental trajectories
sc.tl.paga(adata)
sc.pl.paga(adata)
adata.uns['iroot'] = np.flatnonzero(adata.obs['cell_type'] == 'stem')[0]
sc.tl.dpt(adata)
```

### Differential Expression

```python
# Find marker genes for each cluster
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=20)
```

Scanpy has revolutionized single-cell biology, enabling researchers to map the cellular diversity of tissues and understand how cells differentiate and function.
