---
name: h5py
description: A Pythonic interface to the HDF5 binary data format. It allows you to store huge amounts of numerical data and easily manipulate that data from NumPy. Features a hierarchical structure similar to a file system. Use for storing datasets larger than RAM, organizing complex scientific data hierarchically, storing numerical arrays with high-speed random access, keeping metadata attached to data, sharing data between languages, and reading/writing large datasets in chunks.
version: 3.10
license: BSD-3-Clause
---

# h5py - Hierarchical Data Storage

h5py provides a seamless bridge between NumPy and HDF5. It allows you to organize data into groups (like folders) and datasets (like NumPy arrays), with rich metadata (attributes) attached to every object.

## When to Use

- Storing datasets that are much larger than your computer's RAM.
- Organizing complex scientific data into a hierarchical "folder-like" structure.
- Storing numerical arrays (NumPy) with high-speed random access.
- Keeping metadata (units, experiment dates, parameters) attached directly to the data.
- Sharing data between different languages (C, C++, Fortran, Java, MATLAB), as HDF5 is a cross-platform standard.
- Reading/writing large datasets in chunks to optimize I/O performance.

## Reference Documentation

**Official docs**: https://docs.h5py.org/  
**HDF Group**: https://www.hdfgroup.org/  
**Search patterns**: `h5py.File`, `create_dataset`, `h5py.Group`, `chunks=True`, `compression="gzip"`

## Core Principles

### The Hierarchy

HDF5 files contain two main types of objects:
- **Datasets**: Multidimensional arrays of data (NumPy-like).
- **Groups**: Container structures that can hold datasets or other groups (like directories).

### Slicing

h5py datasets support standard NumPy slicing. When you slice a dataset, only that specific slice is read from the disk, keeping memory usage low.

### Attributes

Every group and dataset can have attributes (key-value pairs) for metadata.

## Quick Reference

### Installation

```bash
pip install h5py
```

### Standard Imports

```python
import h5py
import numpy as np
```

### Basic Pattern - Writing and Reading

```python
import h5py
import numpy as np

# Writing data
with h5py.File('data.h5', 'w') as f:
    dset = f.create_dataset('main_data', data=np.random.rand(100, 100))
    dset.attrs['units'] = 'meters'
    grp = f.create_group('subgroup')
    grp.create_dataset('results', data=[1, 2, 3])

# Reading data
with h5py.File('data.h5', 'r') as f:
    data_slice = f['main_data'][0:10, 0:10] # Only read 100 elements
    units = f['main_data'].attrs['units']
    print(f"Group content: {list(f['subgroup'].keys())}")
```

## Critical Rules

### ✅ DO

- **Use Context Managers** - Always use `with h5py.File(...) as f:` to ensure files are closed even if errors occur.
- **Use Chunking** - For large datasets, specify `chunks=True` or a manual shape to optimize access speed for specific slicing patterns.
- **Enable Compression** - Use `compression="gzip"` to save disk space for large numerical arrays.
- **Use Descriptive Names** - Use groups to organize data logically (e.g., `/experiment1/sensorA/raw`).
- **Store Metadata in Attributes** - Don't create separate text files for units or timestamps; attach them to the datasets.
- **Check Membership** - Use `"name" in group` before accessing to avoid KeyError.

### ❌ DON'T

- **Open files in 'w' by mistake** - The 'w' mode overwrites existing files. Use 'a' (append/read-write) or 'r+' (read-write) instead.
- **Load entire datasets into RAM** - Avoid `data = f['large_dataset'][:]` unless you are sure it fits in memory.
- **Store thousands of small datasets** - HDF5 is optimized for large arrays. For millions of tiny scalars, use a single array or a different database.
- **Forget to close files** - An unclosed HDF5 file can become corrupted or locked.

## Anti-Patterns (NEVER)

```python
import h5py
import numpy as np

# ❌ BAD: Manual file closing (unsafe)
f = h5py.File('data.h5', 'w')
f.create_dataset('x', data=np.arange(10))
f.close() # If an error happened above, this never runs!

# ✅ GOOD: Context manager
with h5py.File('data.h5', 'w') as f:
    f.create_dataset('x', data=np.arange(10))

# ❌ BAD: Storing metadata as strings inside a dataset
f.create_dataset('meta', data=np.array(['unit: meter', 'date: 2024']))

# ✅ GOOD: Using Attributes
dset = f.create_dataset('data', data=np.random.rand(10))
dset.attrs['unit'] = 'meter'
dset.attrs['date'] = '2024'

# ❌ BAD: Inefficient chunking (one row at a time when you read columns)
# f.create_dataset('big', shape=(10000, 10000), chunks=(1, 10000))
```

## Dataset Creation and Configuration

### Advanced Options

```python
with h5py.File('optimized.h5', 'w') as f:
    # 1. Resizable dataset (maxshape)
    dset = f.create_dataset('growing', 
                            shape=(100,), 
                            maxshape=(None,), # Allow growth in 1st dimension
                            dtype='float32')
    
    # 2. Compression and Chunking
    f.create_dataset('compressed', 
                     data=np.random.randn(1000, 1000),
                     chunks=(100, 100), 
                     compression="gzip", 
                     compression_opts=4) # 4 is a good balance
    
    # 3. Filling with default values
    f.create_dataset('default', shape=(10, 10), fillvalue=-1.0)
```

## Working with Groups

### Navigation and Iteration

```python
with h5py.File('nested.h5', 'w') as f:
    f.create_group('raw/2024/january')
    f.create_group('raw/2024/february')

# Recursive iteration
def print_structure(name, obj):
    print(name)

with h5py.File('nested.h5', 'r') as f:
    f.visititems(print_structure) # Visits every dataset and group

# Accessing via path
feb_data = f['/raw/2024/february']
```

## Performance Optimization

### 1. Chunking Strategies

Chunks are the smallest unit of data that can be read or written.
- If you usually read row by row: `chunks=(1, n_cols)`.
- If you read blocks: `chunks=(100, 100)`.
- If unsure: `chunks=True` lets h5py guess.

### 2. SWMR (Single Writer Multiple Reader)

Allows a writer to append to a file while other processes read from it in real-time.

```python
# Writer
f = h5py.File('live.h5', 'w', libver='latest')
f.swmr_mode = True

# Reader
f = h5py.File('live.h5', 'r', libver='latest', swmr=True)
```

### 3. Core Driver (In-Memory HDF5)

Use HDF5 structure but keep it entirely in RAM for speed, with optional save to disk.

```python
# Create an HDF5 file in memory
f = h5py.File('memfile.h5', 'w', driver='core', backing_store=True)
```

## Practical Workflows

### 1. Storing Machine Learning Training Data

```python
def save_ml_dataset(X, y, filename):
    with h5py.File(filename, 'w') as f:
        # Create datasets for images and labels
        f.create_dataset('images', data=X, compression="lzf") # LZF is fast
        f.create_dataset('labels', data=y)
        
        # Add metadata
        f.attrs['n_samples'] = X.shape[0]
        f.attrs['input_shape'] = X.shape[1:]
        f.attrs['classes'] = np.unique(y)

# Use cases: training on data that exceeds RAM
```

### 2. Large Simulation Logger

```python
def log_simulation_step(filename, step_idx, data_array):
    with h5py.File(filename, 'a') as f:
        if 'simulation' not in f:
            # Initialize resizable dataset
            f.create_dataset('simulation', 
                            shape=(0, *data_array.shape),
                            maxshape=(None, *data_array.shape),
                            chunks=(1, *data_array.shape))
            
        dset = f['simulation']
        dset.resize(step_idx + 1, axis=0)
        dset[step_idx] = data_array
```

### 3. Batch Image Storage

```python
def store_images(image_files, h5_file):
    with h5py.File(h5_file, 'w') as f:
        grp = f.create_group('microscopy_data')
        for i, img_path in enumerate(image_files):
            # Load your image here
            img_data = np.random.rand(512, 512) 
            dset = grp.create_dataset(f'img_{i:04d}', data=img_data)
            dset.attrs['original_path'] = img_path
```

## Common Pitfalls and Solutions

### The "Dataset Already Exists" Error

```python
# ❌ Problem: f.create_dataset('x', ...) fails if 'x' exists
# ✅ Solution: Delete first or use a check
if 'x' in f:
    del f['x']
f.create_dataset('x', data=new_data)
```

### File Locking Issues

```python
# ❌ Problem: "OSError: Unable to open file (file locking disabled on this file system)"
# This often happens on network drives (NFS).

# ✅ Solution: Set environment variable before running script
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
```

### Storing Unicode Strings

HDF5's support for strings is complex.

```python
# ❌ Problem: Storing lists of strings can sometimes cause issues in older versions
# ✅ Solution: Use special string types
dt = h5py.string_dtype(encoding='utf-8')
dset = f.create_dataset('strings', (100,), dtype=dt)
dset[0] = "Научные данные"
```

h5py is the industrial-strength way to handle large numerical data. By combining the flexibility of NumPy with the power of HDF5, it ensures that your scientific data remains organized, accessible, and fast.
