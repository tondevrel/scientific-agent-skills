---
name: scikit-bio
description: Library for bioinformatics and community ecology statistics. Provides data structures and algorithms for sequences, alignments, phylogenetics, and diversity analysis. Essential for microbiome research and ecological data science. Use for alpha/beta diversity metrics, ordination (PCoA), phylogenetic trees, sequence manipulation (DNA/RNA/Protein), distance matrices, PERMANOVA, and community ecology analysis.
version: 0.6
license: BSD-3-Clause
---

# scikit-bio - Bioinformatics and Ecology

scikit-bio provides the data structures and statistical methods needed for rigorous biological analysis. It excels in calculating alpha/beta diversity, performing ordination (PCoA), and handling complex phylogenetic trees.

## When to Use

- Analyzing microbiome data (taxonomic composition, community structure).
- Calculating ecological diversity metrics (Shannon, Simpson, UniFrac).
- Performing ordination for visualization (PCoA, DCA).
- High-level sequence manipulation (DNA, RNA, Protein with metadata).
- Reading and writing phylogenetic trees (Newick format).
- Pairwise and multiple sequence alignment analysis.
- Statistical testing of community differences (PERMANOVA, ANOSIM).

## Reference Documentation

**Official docs**: http://scikit-bio.org/  
**Tutorials**: http://scikit-bio.org/docs/latest/  
**Search patterns**: `skbio.sequence`, `skbio.stats.distance`, `skbio.diversity`, `skbio.stats.ordination`

## Core Principles

### Grammar of Biological Sequences

Instead of using raw strings, scikit-bio uses typed objects (DNA, RNA, Protein). These objects know their alphabet, can handle quality scores (Phred), and support biological operations (transcription, translation).

### Distance Matrices

Microbiome research often boils down to comparing samples. The DistanceMatrix object is central, allowing for easy indexing, sub-setting, and statistical testing (e.g., PERMANOVA).

### Diversity Metrics

Provides a standardized implementation of hundreds of diversity metrics used in ecology, ensuring reproducibility across studies.

## Quick Reference

### Installation

```bash
pip install scikit-bio
```

### Standard Imports

```python
import skbio
import numpy as np
import pandas as pd
from skbio import DNA, RNA, Protein, Sequence
from skbio.stats.distance import DistanceMatrix
from skbio.diversity import alpha, beta
from skbio.stats.ordination import pcoa
```

### Basic Pattern - Sequence Manipulation

```python
from skbio import DNA

# 1. Create a sequence with metadata
seq = DNA("ACC--GTT", metadata={'id': 'sample1', 'desc': 'gene A'})

# 2. Biological operations
rc = seq.reverse_complement()
degapped = seq.degap()

# 3. Validation
print(f"Is valid? {seq.is_valid()}") # Checks against DNA alphabet
```

## Critical Rules

### ✅ DO

- **Use Typed Sequences** - Always use DNA, RNA, or Protein instead of generic Sequence to enable alphabet-specific methods.
- **Set Metadata** - Use the metadata and positional_metadata (for quality scores) attributes to keep data self-contained.
- **Check Alphabet** - Use `.is_valid()` when importing data from untrusted sources to catch non-IUPAC characters.
- **Project Distance Matrices** - Use `pcoa()` to visualize high-dimensional community data.
- **Specify Reference for UniFrac** - When calculating UniFrac diversity, ensure your phylogenetic tree contains all taxa present in your abundance table.
- **Validate Trees** - Use `skbio.tree.TreeNode` for tree manipulations as it provides robust traversal methods.

### ❌ DON'T

- **Mix Sequence Types** - Don't try to align DNA with RNA objects.
- **Ignore Quality Scores** - If you have FASTQ data, store Phred scores in positional_metadata.
- **Manually Calculate Distance** - Don't use raw NumPy for biological distances; use `skbio.diversity.beta` to access UniFrac or Bray-Curtis.
- **Forget ID Matching** - When performing ordination or PERMANOVA, ensure IDs in your distance matrix and metadata mapping match exactly.

## Anti-Patterns (NEVER)

```python
from skbio import DNA

# ❌ BAD: Manual reverse complement with string logic
rc_str = seq_str[::-1].replace('A', 't').replace('T', 'a')... # Fragile

# ✅ GOOD: Built-in validated method
rc_seq = DNA(seq_str).reverse_complement()

# ❌ BAD: Using raw lists for community analysis
# data = [[1, 0, 5], [2, 1, 0]]
# dist = manual_bray_curtis(data)

# ✅ GOOD: Using DistanceMatrix
from skbio.stats.distance import DistanceMatrix
dm = DistanceMatrix(matrix_data, ids=['S1', 'S2', 'S3'])

# ❌ BAD: Stripping metadata to run scikit-learn
# vals = seq.values # Metadata lost!

# ✅ GOOD: Process within skbio or use standard IO
seq.write('output.fasta')
```

## Sequence Analysis (skbio.sequence)

### Advanced Sequence Operations

```python
from skbio import DNA, Protein

# Sequence with quality scores
seq = DNA("ACGT", positional_metadata={'quality': [30, 35, 40, 20]})

# Slicing preserves metadata
sub = seq[1:3] 
print(sub.positional_metadata) # {'quality': [35, 40]}

# K-mer frequencies
freqs = seq.kmer_frequencies(k=2)

# Translation (DNA -> Protein)
protein = DNA("ATGCGA").translate()
```

## Diversity Analysis (skbio.diversity)

### Alpha Diversity (Within a sample)

```python
from skbio.diversity import alpha

counts = [10, 0, 5, 2, 20] # Species abundances
otus = ['OTU1', 'OTU2', 'OTU3', 'OTU4', 'OTU5']

shannon = alpha.shannon(counts)
simpson = alpha.simpson(counts)
observed_otus = alpha.observed_otus(counts)

print(f"Shannon Index: {shannon:.3f}")
```

### Beta Diversity (Between samples)

```python
from skbio.diversity import beta
import numpy as np

# Abundance table (samples x taxa)
data = np.array([[10, 20, 0], 
                 [5, 15, 2], 
                 [0, 1, 30]])
ids = ['Sample1', 'Sample2', 'Sample3']

# Bray-Curtis distance
bc_dm = beta.pw_distances(data, ids=ids, metric='braycurtis')

# Weighted UniFrac (Requires a tree)
# unifrac_dm = beta.weighted_unifrac(data, otu_ids, tree)
```

## Phylogenetics (skbio.tree)

### Handling Trees

```python
from skbio import TreeNode
from io import StringIO

# Load Newick tree
tree_str = "((A:0.1, B:0.2)C:0.3, D:0.4)E;"
tree = TreeNode.read(StringIO(tree_str))

# Traverse
for node in tree.tips():
    print(f"Leaf: {node.name}, dist: {node.length}")

# Find Lowest Common Ancestor
lca = tree.find_lca(['A', 'B'])
print(f"LCA of A and B is {lca.name}")

# Rooting
tree.root_at('D')
```

## Statistics and Ordination

### PCoA (Principal Coordinates Analysis)

```python
from skbio.stats.ordination import pcoa
from skbio.stats.distance import DistanceMatrix

# Create DM
dm = DistanceMatrix([[0, 0.5, 0.8], [0.5, 0, 0.2], [0.8, 0.2, 0]], 
                    ids=['A', 'B', 'C'])

# Perform PCoA
results = pcoa(dm)

# View proportion of variance explained
print(results.proportion_explained)

# Access coordinates for plotting
coords = results.samples
```

### Community Testing (PERMANOVA)

```python
from skbio.stats.distance import permanova

# metadata linking IDs to groups
metadata = pd.DataFrame({
    'BodySite': ['Gut', 'Gut', 'Skin', 'Skin']}, 
    index=['S1', 'S2', 'S3', 'S4'])

# Assuming dm is a DistanceMatrix of S1..S4
# results = permanova(dm, metadata, column='BodySite', permutations=999)
# print(results['p-value'])
```

## Practical Workflows

### 1. Microbiome Distance Visualization

```python
def plot_microbiome_pcoa(abundance_df, metadata_df, group_col):
    """Full workflow: Abundance -> Distance -> PCoA -> Table."""
    from skbio.diversity import beta
    from skbio.stats.ordination import pcoa
    
    # 1. Calculate Bray-Curtis distance
    dm = beta.pw_distances(abundance_df.values, ids=abundance_df.index, metric='braycurtis')
    
    # 2. PCoA
    pc = pcoa(dm)
    
    # 3. Merge with metadata for plotting
    plot_data = pc.samples[['PC1', 'PC2']].join(metadata_df)
    
    return plot_data # Ready for Seaborn/Plotly
```

### 2. Sequence QC and Filtering

```python
def filter_low_quality_sequences(sequences, min_avg_qual=30):
    """Filters DNA sequences based on Phred scores."""
    valid_seqs = []
    for s in sequences:
        # Assuming quality is in positional_metadata
        avg_q = np.mean(s.positional_metadata['quality'])
        if avg_q >= min_avg_qual:
            valid_seqs.append(s)
    return valid_seqs
```

### 3. Phylogenetic Distance Calculation

```python
def get_tip_distances(tree):
    """Returns a distance matrix of all tips in a tree."""
    dm = tree.tip_tip_distances()
    return dm
```

## Performance Optimization

### Vectorized Diversity

When calculating diversity for many samples, pass the entire 2D array to `beta.pw_distances` instead of looping through pairs.

### Tree Traversal

Use `tree.preorder()` or `tree.postorder()` instead of manual recursion for much better performance on large (10,000+ tip) trees.

## Common Pitfalls and Solutions

### The "Missing ID" in Distance Matrix

PERMANOVA and PCoA will fail if your metadata index and DistanceMatrix IDs don't match.

```python
# ✅ Solution: Ensure alignment
common_ids = set(dm.ids).intersection(metadata.index)
dm_sub = dm.filter(common_ids)
metadata_sub = metadata.loc[list(common_ids)]
```

### UniFrac Memory Usage

UniFrac calculation can be memory-intensive. For massive datasets, consider using the `skbio.diversity.beta.unweighted_unifrac` optimized versions or external tools like Unifrac-Binaries.

### IUPAC Ambiguity

Standard DNA objects don't allow arbitrary characters.

```python
# ❌ Problem: DNA("ACGN") raises error if 'N' isn't handled
# ✅ Solution: scikit-bio DNA supports IUPAC (N, R, Y, etc.) by default.
# But if you have non-standard characters:
from skbio import Sequence
custom = Sequence("ACG-X") # Generic sequence permits anything
```

scikit-bio is the mathematical heart of modern microbiome and evolutionary research. By enforcing strict biological typing and providing validated ecological metrics, it ensures that your biological insights are grounded in statistical rigor.
