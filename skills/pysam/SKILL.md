---
name: pysam
description: Python module for reading, manipulating and writing genomic alignment formats (SAM/BAM/CRAM) and variant files (VCF/BCF). Wrapper for htslib.
version: 0.22
license: MIT
---

# Pysam - Genomic Alignments

Used for high-throughput sequencing pipelines. It allows efficient access to billions of DNA fragments aligned to a reference genome.

## When to Use

- Processing next-generation sequencing (NGS) data.
- Analyzing genomic variants (SNPs, indels).
- Extracting reads from specific genomic regions.
- Building custom bioinformatics pipelines.
- Quality control of sequencing data.

## Core Principles

### Indexed Access

BAM files must be indexed (.bai) for efficient random access to genomic regions.

### Coordinate System

Genomic coordinates are 0-based (Python-style) for positions, but 1-based for ranges in some contexts.

### Read Attributes

Each read contains sequence, quality scores, alignment position, and flags.

## Quick Reference

### Standard Imports

```python
import pysam
```

### Basic Patterns

```python
# 1. Open BAM file
samfile = pysam.AlignmentFile("aligned_reads.bam", "rb")

# 2. Iterate over reads in a specific genomic region
for read in samfile.fetch("chr1", 10000, 10100):
    print(f"Read: {read.query_name}, Quality: {read.mapping_quality}")
    print(f"Sequence: {read.query_sequence}")
    print(f"Position: {read.reference_start}")

# 3. Variant analysis (VCF)
vcf = pysam.VariantFile("mutations.vcf")
for rec in vcf.fetch("chr1", 10000, 10100):
    print(f"Pos: {rec.pos}, Ref: {rec.ref}, Alt: {rec.alts}")
    print(f"Genotype: {rec.samples['sample1']['GT']}")

# 4. Writing aligned reads
outfile = pysam.AlignmentFile("output.bam", "wb", template=samfile)
for read in samfile:
    if read.mapping_quality > 30:
        outfile.write(read)
outfile.close()
```

## Critical Rules

### ✅ DO

- **Always use indexed files** - Create index with `pysam.index("file.bam")` for fast access.
- **Check read flags** - Use `read.is_paired`, `read.is_unmapped` to filter reads.
- **Handle unmapped reads** - Unmapped reads have `reference_start = -1`.
- **Close files explicitly** - Use context managers or `.close()` to avoid resource leaks.

### ❌ DON'T

- **Don't iterate over entire BAM** - Use `fetch()` with regions for efficiency.
- **Don't ignore quality scores** - Low-quality bases can cause false variants.
- **Don't mix coordinate systems** - Be consistent with 0-based vs 1-based indexing.

## Advanced Patterns

### Counting Reads per Gene

```python
# Using a gene annotation file
genes = {}  # gene_name -> (chr, start, end)
for read in samfile.fetch():
    # Check if read overlaps any gene
    for gene, (chr, start, end) in genes.items():
        if read.reference_name == chr and start <= read.reference_start < end:
            genes[gene]['count'] += 1
```

### Variant Filtering

```python
# Filter variants by quality and depth
for rec in vcf.fetch():
    depth = rec.samples['sample1']['DP']
    qual = rec.qual
    if depth > 10 and qual > 20:
        # Process high-quality variant
        pass
```

Pysam provides the low-level access needed for genomic data processing, enabling researchers to work directly with the raw data of life itself.
