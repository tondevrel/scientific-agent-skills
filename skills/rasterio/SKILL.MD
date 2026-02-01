---
name: rasterio
description: Raster geospatial data processing — the Python interface to GDAL for satellite imagery, elevation models, and grid-based geographic analysis. Rasterio reads and writes georeferenced raster formats (GeoTIFF, NetCDF, JP2, PNG, JPEG2000), handles Coordinate Reference Systems (CRS) and reprojection, performs band math (NDVI, NDWI, EVI), clips/masks rasters with vector geometries, resamples grids, and supports memory-efficient windowed I/O for multi-gigabyte files. Use when: working with satellite imagery or aerial photos, processing Digital Elevation Models (DEM/DTM/DSM), computing spectral indices from multispectral data, clipping raster data to polygon boundaries, reprojecting between coordinate systems, performing spatial interpolation on gridded data, analyzing land cover or land use change over time, integrating raster data with vector data (geopandas/shapely), or any task involving georeferenced grid/pixel data as opposed to vector points/lines/polygons.
version: 1.3.0
license: BSD-3-Clause
---

# Rasterio — Raster Geospatial Processing

Rasterio is the standard Python library for reading and writing georeferenced raster data. It wraps GDAL but exposes a clean Pythonic API. Every raster has two components: **pixel values** (a numpy array) and **geospatial metadata** (CRS, transform, bounds) that maps pixels to real-world coordinates.

## Raster vs Vector — When to Use What

```
RASTER (Rasterio)                      VECTOR (GeoPandas / Shapely)
────────────────────────────────       ────────────────────────────────
Grid of pixels                         Points, lines, polygons
Satellite imagery                      Administrative boundaries
Elevation models (DEM)                 Road networks
Land cover maps                        Locations / GPS tracks
Temperature grids                      Census regions
Spectral data                          Feature geometries

KEY RULE: When your data IS a grid → Rasterio.
          When your data IS shapes/points → GeoPandas.
          When you need BOTH → Rasterio clips rasters TO vector shapes.
```

## Reference Documentation

**Rasterio docs**: https://rasterio.readthedocs.io/en/latest/  
**GDAL formats**: https://gdal.org/drivers/raster/index.html  
**GitHub**: https://github.com/rasterio/rasterio  
**Search patterns**: `rasterio.open`, `dataset.read`, `dataset.transform`, `rasterio.mask`, `show`

## Core Principles

### The Dataset Object
`rasterio.open()` returns a dataset. A dataset has **bands** (layers of pixel data), a **transform** (maps pixel coordinates to geographic coordinates), a **CRS** (coordinate reference system), and **bounds** (geographic extent). Always use context manager (`with` statement) — it handles file handles correctly.

### The Transform
An affine transform maps pixel (col, row) to geographic (x, y). `dataset.transform` gives this mapping. `dataset.index(x, y)` gives the reverse: geographic → pixel. This is how you go from "latitude/longitude" to "which pixel?"

### Bands
Most satellite imagery is multi-band: Band 1 = Red, Band 2 = Green, Band 3 = Blue, Band 4 = NIR, etc. Band numbering starts at **1** (not 0). `dataset.read(1)` reads band 1 as a 2D numpy array.

### CRS — Coordinate Reference Systems
Every raster is projected into some CRS. `EPSG:4326` = WGS84 lat/lon (GPS coordinates). `EPSG:32633` = UTM zone 33N (meters, good for Europe). Operations between rasters in different CRS require reprojection first.

### NoData
Pixels outside valid coverage are marked with a `nodata` value (e.g., -9999, 0, or NaN). Always mask these before computation — including them corrupts statistics.

## Quick Reference

### Installation

```bash
pip install rasterio numpy matplotlib
# For vector integration:
pip install geopandas shapely fiona
```

### Standard Imports

```python
import rasterio
from rasterio.transform import from_bounds, Affine
from rasterio.crs import CRS
import numpy as np
import matplotlib.pyplot as plt
```

### Basic Pattern — Read, Inspect, Visualize

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open('image.tif') as src:
    # Metadata
    print(f"Bands:      {src.count}")
    print(f"Shape:      {src.height} x {src.width}")
    print(f"CRS:        {src.crs}")
    print(f"Bounds:     {src.bounds}")           # (left, bottom, right, top)
    print(f"Resolution: {src.res}")              # (x_res, y_res) in CRS units
    print(f"NoData:     {src.nodata}")
    print(f"Transform:  {src.transform}")

    # Read all bands → shape: (bands, height, width)
    data = src.read()

    # Read single band → shape: (height, width)
    band1 = src.read(1)

    # Read with masked nodata → numpy masked array
    band1_masked = src.read(1, masked=True)

# Visualize single band
plt.imshow(band1, cmap='viridis')
plt.colorbar(label='Value')
plt.title('Band 1')
plt.show()
```

### Basic Pattern — Write a Raster

```python
import rasterio
from rasterio.transform import from_bounds
import numpy as np

# Create a 100x100 single-band raster covering a geographic extent
height, width = 100, 100
data = np.random.rand(height, width).astype(np.float32)

transform = from_bounds(
    west=10.0, south=50.0, east=11.0, north=51.0,
    width=width, height=height
)

with rasterio.open(
    'output.tif',
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,                          # Number of bands
    dtype=data.dtype,
    crs='EPSG:4326',                  # WGS84
    transform=transform,
    nodata=-9999
) as dst:
    dst.write(data, 1)                # Write to band 1
```

## Critical Rules

### ✅ DO

- **Always use `with rasterio.open(...)` context manager** — Ensures file handles are closed properly. Never do `src = rasterio.open(...)` without `with`.
- **Use `masked=True` when reading** — Returns a numpy masked array that automatically excludes nodata pixels. Prevents nodata from corrupting calculations.
- **Match CRS before any spatial operation** — If two rasters have different CRS, reproject one before combining. Use `rasterio.warp.reproject()`.
- **Use windowed reading for large files** — Files > 1GB will OOM if read entirely. Use `src.read(window=...)` to read chunks.
- **Preserve geospatial metadata when writing** — Copy `transform`, `crs`, `nodata` from the source when creating derived rasters.
- **Use `float32` for computed indices** — NDVI, NDWI produce float values [-1, 1]. Input bands are often `uint8` or `uint16` — cast before division to avoid integer truncation.
- **Check `src.nodata` before computing** — If nodata is None, the file has no declared nodata value. Handle accordingly.
- **Use `rasterio.features` for vector-raster conversion** — Don't roll your own rasterization.

### ❌ DON'T

- **Don't mix up band indexing** — Rasterio bands start at **1**. `src.read(1)` = first band. numpy arrays are 0-indexed, so `data[0]` = first band after `src.read()`.
- **Don't forget to cast dtypes before band math** — `uint8 / uint8` = integer division = truncated to 0. Always cast: `band.astype(np.float32)`.
- **Don't assume all rasters share the same CRS** — Even "standard" datasets may differ. Always check and align.
- **Don't ignore resolution differences** — Two rasters covering the same area may have different pixel sizes. Resample to match before pixel-wise operations.
- **Don't hardcode nodata as 0** — Nodata values vary by file. Always read `src.nodata`.
- **Don't read entire multi-GB files into memory** — Use windowed I/O or overview levels.

## Anti-Patterns (NEVER)

```python
import rasterio
import numpy as np

# ❌ BAD: Integer division in band math — truncates to 0
with rasterio.open('sentinel.tif') as src:
    nir  = src.read(4)    # uint16
    red  = src.read(3)    # uint16
    ndvi = (nir - red) / (nir + red)   # INTEGER DIVISION → all zeros or ±1!

# ✅ GOOD: Cast to float BEFORE any arithmetic
with rasterio.open('sentinel.tif') as src:
    nir  = src.read(4).astype(np.float32)
    red  = src.read(3).astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-10)   # +1e-10 avoids div-by-zero

# ─────────────────────────────────────────────────────────────

# ❌ BAD: No context manager — file handle leak
src  = rasterio.open('image.tif')
data = src.read(1)
# src never closed → resource leak, potential corruption on write

# ✅ GOOD: Context manager
with rasterio.open('image.tif') as src:
    data = src.read(1)
# File closed automatically

# ─────────────────────────────────────────────────────────────

# ❌ BAD: Ignoring nodata in statistics
with rasterio.open('dem.tif') as src:
    elev = src.read(1)
    mean_elevation = elev.mean()    # Includes -9999 nodata → wrong answer!

# ✅ GOOD: Mask nodata before computing
with rasterio.open('dem.tif') as src:
    elev   = src.read(1, masked=True)           # Masked array
    mean_elevation = float(elev.mean())          # Ignores masked (nodata) pixels
    # OR explicitly:
    valid = elev[elev != src.nodata]
    mean_elevation = valid.mean()

# ─────────────────────────────────────────────────────────────

# ❌ BAD: Operating on rasters with different CRS without reprojection
with rasterio.open('landcover_utm.tif') as src1:
    with rasterio.open('population_wgs84.tif') as src2:
        # src1.crs = EPSG:32633 (UTM), src2.crs = EPSG:4326 (WGS84)
        pop  = src2.read(1)
        land = src1.read(1)
        result = pop * land   # WRONG — pixels don't align geographically!

# ✅ GOOD: Reproject to common CRS first (see Reprojection section)
```

## Reading Rasters

### Full Read vs Selective Read

```python
import rasterio
import numpy as np

with rasterio.open('multispectral.tif') as src:
    # All bands at once → shape: (n_bands, height, width)
    all_bands = src.read()

    # Single band → shape: (height, width)
    band1 = src.read(1)

    # Multiple specific bands → shape: (n_selected, height, width)
    rgb = src.read([1, 2, 3])

    # With nodata masking
    band1_masked = src.read(1, masked=True)   # numpy.ma.MaskedArray

    # Specific spatial subset (window)
    from rasterio.windows import Window
    window = Window(col_off=100, row_off=200, width=50, height=50)
    patch  = src.read(1, window=window)       # shape: (50, 50)

    # Read at lower resolution (overview)
    # out_shape forces resampling to smaller array
    small = src.read(1, out_shape=(src.height // 4, src.width // 4))
```

### Coordinate ↔ Pixel Mapping

```python
import rasterio

with rasterio.open('image.tif') as src:
    # Geographic coordinates → pixel (row, col)
    row, col = src.index(lon, lat)       # e.g., src.index(10.5, 50.3)

    # Pixel (row, col) → geographic coordinates (x, y)
    x, y = src.xy(row, col)             # Center of that pixel

    # Geographic extent of a specific pixel
    # Top-left corner of pixel (row, col):
    x_tl = src.transform.c + col * src.transform.a
    y_tl = src.transform.f + row * src.transform.e

    # Bounds of the entire raster
    print(f"Left={src.bounds.left}, Bottom={src.bounds.bottom}, "
          f"Right={src.bounds.right}, Top={src.bounds.top}")
```

## Writing Rasters

### Write with Full Metadata

```python
import rasterio
from rasterio.transform import from_bounds, Affine
import numpy as np

def write_raster(data: np.ndarray,
                 output_path: str,
                 transform: Affine,
                 crs: str = 'EPSG:4326',
                 nodata: float = -9999,
                 band_descriptions: list[str] = None):
    """
    Write a numpy array as a georeferenced GeoTIFF.
    data shape: (bands, height, width) or (height, width) for single band.
    """
    if data.ndim == 2:
        data = data[np.newaxis, :]       # Add band dimension → (1, H, W)

    n_bands, height, width = data.shape

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=n_bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress='lzw',                  # Compression — reduces file size significantly
        tiled=True,                      # Tiled layout — better for large file random access
        blockxsize=256,
        blockysize=256
    ) as dst:
        dst.write(data)
        if band_descriptions:
            for i, desc in enumerate(band_descriptions, 1):
                dst.set_band_description(i, desc)

# ─── Copy metadata pattern (most common: derive new raster from existing) ───
with rasterio.open('input.tif') as src:
    computed = src.read(1).astype(np.float32) * 2.0   # Some computation

    # Copy ALL metadata from source, override only what changed
    profile = src.profile.copy()
    profile.update(dtype=computed.dtype, count=1, nodata=-9999)

    with rasterio.open('output.tif', 'w', **profile) as dst:
        dst.write(computed, 1)
```

## CRS and Reprojection

### Check and Compare CRS

```python
import rasterio
from rasterio.crs import CRS

with rasterio.open('image.tif') as src:
    print(f"CRS: {src.crs}")
    print(f"EPSG: {src.crs.to_epsg()}")
    print(f"WKT:  {src.crs.to_wkt()}")

    # Check if two CRS are the same
    target_crs = CRS.from_epsg(32633)   # UTM zone 33N
    print(f"Same as UTM 33N? {src.crs == target_crs}")

# Common EPSG codes:
# EPSG:4326  — WGS84 lat/lon (GPS, most web maps)
# EPSG:3857  — Web Mercator (Google Maps, OpenStreetMap tiles)
# EPSG:32601–32660 — UTM zones 1–60 North (meters, good for local analysis)
# EPSG:32701–32760 — UTM zones 1–60 South
```

### Reproject a Raster

```python
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import numpy as np

def reproject_raster(input_path: str, output_path: str, target_crs: str = 'EPSG:4326'):
    """Reproject an entire raster file to a new CRS."""
    target_crs = CRS.from_user_input(target_crs)

    with rasterio.open(input_path) as src:
        # Calculate the optimal transform and dimensions for the target CRS
        transform, width, height = calculate_default_transform(
            src.crs, target_crs,
            src.width, src.height,
            *src.bounds
        )

        # Prepare output profile
        profile = src.profile.copy()
        profile.update(crs=target_crs, transform=transform, width=width, height=height)

        with rasterio.open(output_path, 'w', **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear   # bilinear for continuous data
                    # Use Resampling.nearest for categorical (land cover class IDs)
                )

# reproject_raster('utm_image.tif', 'wgs84_image.tif', 'EPSG:4326')
```

## Band Math — Spectral Indices

```python
import rasterio
import numpy as np

def compute_indices(input_path: str, output_path: str,
                    band_map: dict = None) -> dict:
    """
    Compute standard spectral indices from multispectral imagery.

    band_map: maps index name to band number.
    Sentinel-2 default: {'blue': 2, 'green': 3, 'red': 4, 'nir': 8, 'swir1': 11, 'swir2': 12}
    Landsat 8 default:  {'blue': 2, 'green': 3, 'red': 4, 'nir': 5, 'swir1': 6, 'swir2': 7}
    """
    if band_map is None:
        band_map = {'blue': 2, 'green': 3, 'red': 4, 'nir': 8, 'swir1': 11, 'swir2': 12}

    with rasterio.open(input_path) as src:
        # Read required bands as float32
        bands = {}
        for name, idx in band_map.items():
            bands[name] = src.read(idx, masked=True).astype(np.float32)

        eps = 1e-10   # Avoid division by zero

        # ─── NDVI: Normalized Difference Vegetation Index ───
        # Range: [-1, 1]. Vegetation > 0.3. Bare soil ≈ 0. Water < 0.
        ndvi = (bands['nir'] - bands['red']) / (bands['nir'] + bands['red'] + eps)

        # ─── NDWI: Normalized Difference Water Index ───
        # Range: [-1, 1]. Water > 0. Vegetation < 0.
        ndwi = (bands['green'] - bands['nir']) / (bands['green'] + bands['nir'] + eps)

        # ─── NDSI: Normalized Difference Snow Index ───
        ndsi = (bands['green'] - bands['swir1']) / (bands['green'] + bands['swir1'] + eps)

        # ─── EVI: Enhanced Vegetation Index ───
        # More robust than NDVI in high-biomass areas
        evi = 2.5 * (bands['nir'] - bands['red']) / (
              bands['nir'] + 6 * bands['red'] - 7.5 * bands['blue'] + 1 + eps)

        # ─── SAVI: Soil Adjusted Vegetation Index ───
        L = 0.5   # Soil brightness correction factor
        savi = ((bands['nir'] - bands['red']) / (bands['nir'] + bands['red'] + L)) * (1 + L)

        # ─── NBR: Normalized Burn Ratio ───
        nbr = (bands['nir'] - bands['swir2']) / (bands['nir'] + bands['swir2'] + eps)

        indices = {'ndvi': ndvi, 'ndwi': ndwi, 'ndsi': ndsi,
                   'evi': evi, 'savi': savi, 'nbr': nbr}

        # Write all indices to a multi-band output
        n_indices = len(indices)
        profile = src.profile.copy()
        profile.update(count=n_indices, dtype='float32', nodata=-9999)

        with rasterio.open(output_path, 'w', **profile) as dst:
            for i, (name, arr) in enumerate(indices.items(), 1):
                dst.write(arr.filled(-9999), i)
                dst.set_band_description(i, name.upper())

    return {name: arr for name, arr in indices.items()}

# indices = compute_indices('sentinel2_scene.tif', 'indices.tif')
```

## Clipping and Masking with Vector Data

```python
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from shapely.geometry import mapping

def clip_raster_to_polygon(raster_path: str,
                           vector_path: str,
                           output_path: str,
                           all_touched: bool = False):
    """
    Clip a raster to the bounding geometry of a vector file.
    Pixels outside the polygon are set to nodata.
    """
    # Load vector geometries
    gdf = gpd.read_file(vector_path)

    with rasterio.open(raster_path) as src:
        # Ensure vector and raster share the same CRS
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # Convert geometries to the format rasterio expects
        geometries = [mapping(geom) for geom in gdf.geometry if geom is not None]

        # Clip: returns (array, transform) for the cropped extent
        out_image, out_transform = mask(
            src,
            geometries,
            crop=True,                  # Shrink extent to bounding box of geometries
            all_touched=all_touched,    # True: include pixels that touch the polygon edge
            nodata=src.nodata if src.nodata else -9999
        )

        # Write clipped raster
        profile = src.profile.copy()
        profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform
        )
        if src.nodata is None:
            profile.update(nodata=-9999)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(out_image)

    return out_image, out_transform

# clip_raster_to_polygon('satellite.tif', 'study_area.shp', 'clipped.tif')

# ─── Zonal Statistics: mean/std/count per polygon ───
def zonal_stats(raster_path: str, vector_path: str, band: int = 1) -> gpd.GeoDataFrame:
    """Compute statistics of raster values within each polygon."""
    gdf = gpd.read_file(vector_path).copy()

    with rasterio.open(raster_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        stats = []
        for idx, row in gdf.iterrows():
            geom = [mapping(row.geometry)]
            try:
                out_image, _ = mask(src, geom, crop=True, all_touched=True)
                values = out_image[band - 1]
                nodata = src.nodata if src.nodata else -9999
                valid  = values[values != nodata]

                stats.append({
                    'count': len(valid),
                    'mean':  float(np.mean(valid)) if len(valid) > 0 else np.nan,
                    'std':   float(np.std(valid))  if len(valid) > 0 else np.nan,
                    'min':   float(np.min(valid))  if len(valid) > 0 else np.nan,
                    'max':   float(np.max(valid))  if len(valid) > 0 else np.nan,
                    'sum':   float(np.sum(valid))  if len(valid) > 0 else np.nan,
                })
            except Exception:
                stats.append({'count': 0, 'mean': np.nan, 'std': np.nan,
                              'min': np.nan, 'max': np.nan, 'sum': np.nan})

        stats_df = gpd.pd.DataFrame(stats)
        return gpd.pd.concat([gdf.reset_index(drop=True), stats_df], axis=1)

# stats = zonal_stats('ndvi.tif', 'municipalities.shp')
# print(stats[['name', 'mean', 'std', 'count']])
```

## Resampling

```python
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
import numpy as np

def resample_to_resolution(input_path: str,
                           output_path: str,
                           target_res: float,
                           resampling: Resampling = Resampling.bilinear):
    """
    Resample a raster to a specific pixel resolution.
    target_res: desired pixel size in CRS units (meters for UTM, degrees for WGS84)

    Resampling choices:
        Resampling.nearest    → categorical data (land cover class IDs)
        Resampling.bilinear   → continuous data (temperature, elevation)
        Resampling.cubic      → smooth continuous data (best quality, slower)
        Resampling.average    → downsampling continuous data (anti-aliasing)
    """
    with rasterio.open(input_path) as src:
        # Calculate new dimensions
        scale_x = src.res[0] / target_res
        scale_y = src.res[1] / target_res
        new_width  = int(src.width  * scale_x)
        new_height = int(src.height * scale_y)

        # New transform at target resolution
        new_transform = rasterio.Affine(
            target_res, 0, src.transform.c,
            0, -target_res, src.transform.f
        )

        profile = src.profile.copy()
        profile.update(width=new_width, height=new_height, transform=new_transform)

        with rasterio.open(output_path, 'w', **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,
                    resampling=resampling
                )

# resample_to_resolution('10m_sentinel.tif', '30m_resampled.tif', target_res=30)


# ─── Resample to match another raster (align grids) ───
def resample_to_match(source_path: str, reference_path: str, output_path: str):
    """Resample source raster to exactly match reference raster's grid."""
    with rasterio.open(reference_path) as ref:
        target_transform = ref.transform
        target_crs       = ref.crs
        target_width     = ref.width
        target_height    = ref.height

    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        profile.update(width=target_width, height=target_height,
                       transform=target_transform, crs=target_crs)

        with rasterio.open(output_path, 'w', **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )

# resample_to_match('population.tif', 'landcover.tif', 'population_aligned.tif')
```

## Windowed I/O — Memory-Efficient Processing

```python
import rasterio
from rasterio.windows import Window
import numpy as np

def process_large_raster(input_path: str, output_path: str,
                         block_size: int = 256):
    """
    Process a large raster tile-by-tile without loading into memory.
    Reads in blocks, applies a function, writes results.
    """
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(dtype='float32')

        with rasterio.open(output_path, 'w', **profile) as dst:
            # Iterate over block windows
            for ji, window in src.block_windows(1):
                # Read this block (all bands)
                data = src.read(window=window).astype(np.float32)

                # ── YOUR COMPUTATION HERE ──
                # Example: normalize each band to [0, 1]
                for b in range(data.shape[0]):
                    band = data[b]
                    valid = band[band != src.nodata] if src.nodata else band
                    if len(valid) > 0:
                        vmin, vmax = valid.min(), valid.max()
                        if vmax > vmin:
                            data[b] = (band - vmin) / (vmax - vmin)
                # ── END COMPUTATION ──

                dst.write(data, window=window)

            print(f"Processed {src.width}x{src.height} in {block_size}x{block_size} blocks")

# process_large_raster('huge_satellite.tif', 'normalized.tif')


# ─── Read specific geographic region without loading full file ───
def read_by_bounds(raster_path: str, bounds: tuple) -> tuple[np.ndarray, dict]:
    """
    Read only the pixels within geographic bounds.
    bounds: (west, south, east, north) in raster's CRS.
    Returns: (array, metadata_dict)
    """
    west, south, east, north = bounds

    with rasterio.open(raster_path) as src:
        # Convert geographic bounds to pixel window
        row_min, col_min = src.index(west, north)   # top-left pixel
        row_max, col_max = src.index(east, south)   # bottom-right pixel

        # Clamp to raster extent
        row_min = max(0, row_min)
        col_min = max(0, col_min)
        row_max = min(src.height, row_max)
        col_max = min(src.width, col_max)

        window = Window(col_min, row_min, col_max - col_min, row_max - row_min)
        data   = src.read(window=window)

        # Transform for this window
        win_transform = src.window_transform(window)

        return data, {
            'transform': win_transform,
            'crs':       src.crs,
            'nodata':    src.nodata,
            'window':    window
        }

# data, meta = read_by_bounds('europe.tif', (10.0, 47.0, 12.0, 49.0))
```

## Visualization

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_raster(raster_path: str, band: int = 1,
                cmap: str = 'viridis', title: str = None):
    """Plot a single band with geographic axes and colorbar."""
    with rasterio.open(raster_path) as src:
        data = src.read(band, masked=True).astype(np.float32)
        bounds = src.bounds

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        data,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin='upper',
        cmap=cmap,
        aspect='equal'
    )
    plt.colorbar(im, ax=ax, label='Value')
    ax.set_xlabel('Longitude' if 'longlat' in str(src.crs) else 'X (m)')
    ax.set_ylabel('Latitude'  if 'longlat' in str(src.crs) else 'Y (m)')
    ax.set_title(title or f'Band {band}')
    plt.tight_layout()
    plt.show()


def plot_rgb(raster_path: str,
             r_band: int = 1, g_band: int = 2, b_band: int = 3,
             stretch: str = 'percentile'):
    """
    Plot a true-color (RGB) composite.
    stretch: 'percentile' (robust, handles outliers) or 'minmax' or None
    """
    with rasterio.open(raster_path) as src:
        r = src.read(r_band).astype(np.float32)
        g = src.read(g_band).astype(np.float32)
        b = src.read(b_band).astype(np.float32)
        bounds = src.bounds

    rgb = np.stack([r, g, b], axis=-1)   # Shape: (H, W, 3)

    if stretch == 'percentile':
        # Per-band percentile stretch — standard for satellite imagery display
        for i in range(3):
            p2, p98 = np.percentile(rgb[:, :, i][rgb[:, :, i] > 0], (2, 98))
            rgb[:, :, i] = np.clip((rgb[:, :, i] - p2) / (p98 - p2 + 1e-10), 0, 1)
    elif stretch == 'minmax':
        for i in range(3):
            vmin, vmax = rgb[:, :, i].min(), rgb[:, :, i].max()
            rgb[:, :, i] = (rgb[:, :, i] - vmin) / (vmax - vmin + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
              origin='upper', aspect='equal')
    ax.set_title('RGB Composite')
    plt.tight_layout()
    plt.show()

# plot_raster('ndvi.tif', cmap='RdYlGn', title='NDVI')
# plot_rgb('sentinel2.tif', r_band=4, g_band=3, b_band=2)
```

## Practical Workflows

### 1. Land Cover Change Detection

```python
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def change_detection(path_t1: str, path_t2: str, band: int = 1) -> dict:
    """
    Detect change between two time periods.
    Both rasters must share the same CRS, resolution, and extent.
    Returns change map and statistics.
    """
    with rasterio.open(path_t1) as src1:
        data_t1  = src1.read(band, masked=True).astype(np.float32)
        nodata   = src1.nodata
        profile  = src1.profile.copy()

    with rasterio.open(path_t2) as src2:
        data_t2  = src2.read(band, masked=True).astype(np.float32)

    # Difference map
    change = data_t2 - data_t1

    # Valid pixels only (both periods have data)
    valid  = ~(data_t1.mask | data_t2.mask) if hasattr(data_t1, 'mask') else np.ones_like(change, dtype=bool)
    change_valid = change[valid]

    # Thresholded change classes
    threshold = 0.1   # Adjust based on index scale
    change_class = np.zeros_like(change, dtype=np.int8)
    change_class[change > threshold]   = 1   # Increase (e.g., more vegetation)
    change_class[change < -threshold]  = -1  # Decrease
    # 0 = no significant change

    # Statistics
    stats = {
        'mean_change':     float(change_valid.mean()),
        'std_change':      float(change_valid.std()),
        'pct_increase':    float((change_class[valid] == 1).mean() * 100),
        'pct_decrease':    float((change_class[valid] == -1).mean() * 100),
        'pct_stable':      float((change_class[valid] == 0).mean() * 100),
        'max_increase':    float(change_valid.max()),
        'max_decrease':    float(change_valid.min()),
    }

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(data_t1,     cmap='RdYlGn'); axes[0].set_title('T1')
    axes[1].imshow(data_t2,     cmap='RdYlGn'); axes[1].set_title('T2')
    axes[2].imshow(change_class, cmap='RdBu',  vmin=-1, vmax=1)
    axes[2].set_title('Change: Blue=decrease, Red=increase')
    for ax in axes: ax.axis('off')
    plt.tight_layout(); plt.show()

    return stats, change, change_class

# stats, change_map, change_classes = change_detection('ndvi_2020.tif', 'ndvi_2023.tif')
# print(pd.DataFrame([stats]).T)
```

### 2. DEM Analysis — Terrain Derivatives

```python
import rasterio
import numpy as np
from scipy.ndimage import uniform_filter

def dem_analysis(dem_path: str, output_prefix: str = 'terrain'):
    """
    Compute terrain derivatives from a Digital Elevation Model:
    → Slope, Aspect, Hillshade, Curvature
    """
    with rasterio.open(dem_path) as src:
        elev     = src.read(1, masked=True).astype(np.float64)
        res_x, res_y = src.res
        profile  = src.profile.copy()
        profile.update(dtype='float32', count=1)

    # ─── Sobel gradients (dz/dx, dz/dy) ───
    # Using central differences
    dz_dx = np.zeros_like(elev)
    dz_dy = np.zeros_like(elev)
    dz_dx[:, 1:-1] = (elev[:, 2:] - elev[:, :-2]) / (2 * res_x)
    dz_dy[1:-1, :] = (elev[2:, :] - elev[:-2, :]) / (2 * res_y)

    # ─── SLOPE: angle of steepest descent (degrees) ───
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # ─── ASPECT: direction of steepest descent (degrees, 0=North, clockwise) ───
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (90 - aspect_deg) % 360   # Convert to compass bearing
    aspect_deg = aspect_deg.astype(np.float32)

    # ─── HILLSHADE: simulated illumination ───
    # Standard parameters: sun azimuth=315°, altitude=45°
    az_rad   = np.radians(315)
    alt_rad  = np.radians(45)
    hillshade = (
        np.sin(alt_rad) * np.cos(slope_rad) +
        np.cos(alt_rad) * np.sin(slope_rad) *
        np.cos(az_rad - np.radians(aspect_deg))
    )
    hillshade = np.clip(hillshade * 255, 0, 255).astype(np.float32)

    # ─── Write outputs ───
    outputs = {
        'slope':     slope_deg,
        'aspect':    aspect_deg,
        'hillshade': hillshade,
    }

    for name, arr in outputs.items():
        with rasterio.open(f'{output_prefix}_{name}.tif', 'w', **profile) as dst:
            dst.write(arr, 1)

    # Plot all derivatives
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cmaps = ['gray', 'RdYlGn', 'hsv', 'gray']
    titles = ['Elevation', 'Slope (°)', 'Aspect (°)', 'Hillshade']
    arrays = [elev, slope_deg, aspect_deg, hillshade]

    for ax, arr, cmap, title in zip(axes.flat, arrays, cmaps, titles):
        im = ax.imshow(arr, cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout(); plt.show()

    return outputs

# outputs = dem_analysis('dem_alps.tif', output_prefix='alps_terrain')
```

### 3. Full Remote Sensing Pipeline

```python
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, calculate_default_transform, Resampling
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

def remote_sensing_pipeline(imagery_path: str,
                            roi_path: str,
                            band_map: dict,
                            target_crs: str = None) -> pd.DataFrame:
    """
    Full pipeline: load → align CRS → clip to ROI → compute indices → zonal stats.

    imagery_path: multispectral satellite image
    roi_path:     vector file with study area polygons (must have 'name' column)
    band_map:     {'red': 3, 'green': 2, 'nir': 4, ...}
    target_crs:   reproject to this CRS (None = use imagery CRS)
    """
    import tempfile, os

    # 1. Load ROI
    roi = gpd.read_file(roi_path)

    # 2. Open imagery and align CRS
    with rasterio.open(imagery_path) as src:
        img_crs = src.crs
        if target_crs:
            from rasterio.crs import CRS
            target = CRS.from_user_input(target_crs)
        else:
            target = img_crs

        roi = roi.to_crs(target)

        # 3. Reproject imagery if needed
        if target != img_crs:
            transform, width, height = calculate_default_transform(
                img_crs, target, src.width, src.height, *src.bounds)
            profile = src.profile.copy()
            profile.update(crs=target, transform=transform, width=width, height=height)

            reprojected_path = tempfile.mktemp(suffix='.tif')
            with rasterio.open(reprojected_path, 'w', **profile) as dst:
                for b in range(1, src.count + 1):
                    reproject(rasterio.band(src, b), rasterio.band(dst, b),
                              src_transform=src.transform, src_crs=img_crs,
                              dst_transform=transform, dst_crs=target,
                              resampling=Resampling.bilinear)
            src_path = reprojected_path
        else:
            src_path = imagery_path

    # 4. Compute indices and zonal stats per polygon
    results = []
    with rasterio.open(src_path) as src:
        eps = 1e-10
        for _, row in roi.iterrows():
            geom = [mapping(row.geometry)]
            try:
                out_img, _ = mask(src, geom, crop=True, all_touched=True)
                # Read bands as float
                bands = {}
                for name, idx in band_map.items():
                    b = out_img[idx - 1].astype(np.float32)
                    nodata = src.nodata if src.nodata else -9999
                    b[b == nodata] = np.nan
                    bands[name] = b

                # Compute NDVI
                if 'nir' in bands and 'red' in bands:
                    ndvi = (bands['nir'] - bands['red']) / (bands['nir'] + bands['red'] + eps)
                    valid = ndvi[~np.isnan(ndvi)]
                    results.append({
                        'name':        row.get('name', 'unknown'),
                        'ndvi_mean':   float(np.nanmean(valid)),
                        'ndvi_std':    float(np.nanstd(valid)),
                        'ndvi_min':    float(np.nanmin(valid)),
                        'ndvi_max':    float(np.nanmax(valid)),
                        'pixel_count': int(len(valid)),
                    })
            except Exception as e:
                results.append({'name': row.get('name', 'unknown'), 'error': str(e)})

    # Cleanup temp file
    if target_crs and os.path.exists(src_path):
        os.remove(src_path)

    return pd.DataFrame(results)

# df = remote_sensing_pipeline(
#     'sentinel2_scene.tif',
#     'municipalities.shp',
#     band_map={'blue': 2, 'green': 3, 'red': 4, 'nir': 8},
#     target_crs='EPSG:32633'
# )
# print(df.sort_values('ndvi_mean', ascending=False))
```

### 4. Raster-to-Vector and Vector-to-Raster

```python
import rasterio
from rasterio.features import shapes, rasterize
import numpy as np
import geopandas as gpd
from shapely.geometry import shape

# ─── RASTER → VECTOR: Extract polygons from classified raster ───
def raster_to_polygons(raster_path: str, band: int = 1) -> gpd.GeoDataFrame:
    """Convert contiguous regions of same value into polygons."""
    with rasterio.open(raster_path) as src:
        data    = src.read(band).astype(np.int32)
        nodata  = int(src.nodata) if src.nodata else -9999
        transform = src.transform
        crs     = src.crs

    # Extract shapes: yields (geometry, value) for each contiguous region
    mask_valid = data != nodata
    polygons = []
    for geom, value in shapes(data, mask=mask_valid, transform=transform):
        polygons.append({
            'geometry': shape(geom),
            'class_id': int(value)
        })

    return gpd.GeoDataFrame(polygons, crs=crs)

# polygons = raster_to_polygons('landcover_classified.tif')
# polygons.to_file('landcover_polygons.shp')


# ─── VECTOR → RASTER: Burn vector attributes into a raster grid ───
def vector_to_raster(vector_path: str,
                     output_path: str,
                     attribute: str,
                     reference_raster: str,
                     fill: float = 0) -> None:
    """
    Rasterize a vector layer onto the grid of a reference raster.
    Each pixel gets the value of the 'attribute' column from the vector.
    """
    gdf = gpd.read_file(vector_path)

    with rasterio.open(reference_raster) as ref:
        gdf = gdf.to_crs(ref.crs)
        transform = ref.transform
        width, height = ref.width, ref.height
        crs = ref.crs

    # Build (geometry, value) pairs for rasterize
    geom_value_pairs = [
        (mapping(row.geometry), row[attribute])
        for _, row in gdf.iterrows()
        if row.geometry is not None
    ]

    # Rasterize
    burned = rasterize(
        geom_value_pairs,
        out_shape=(height, width),
        transform=transform,
        fill=fill,
        dtype=np.float32,
        all_touched=False
    )

    # Write
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=height, width=width, count=1,
        dtype=burned.dtype, crs=crs, transform=transform, nodata=fill
    ) as dst:
        dst.write(burned, 1)

# vector_to_raster('population.shp', 'pop_raster.tif',
#                  attribute='pop_density', reference_raster='dem.tif')
```

## Common Pitfalls and Solutions

### Y-Axis Flipped in Plots

```python
import rasterio
import matplotlib.pyplot as plt

# ❌ PROBLEM: imshow default shows row 0 at top (correct for images,
#    but confusing if you also show geographic coordinates)
with rasterio.open('dem.tif') as src:
    data = src.read(1)
    plt.imshow(data)   # Row 0 at top — matches raster convention, but
                       # if extent is set, latitude axis is inverted

# ✅ GOOD: Use origin='upper' (default for rasters) with extent
with rasterio.open('dem.tif') as src:
    data   = src.read(1)
    bounds = src.bounds
    plt.imshow(data,
               extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
               origin='upper')    # Row 0 = top = north. Correct for geographic display.
```

### Nodata Corrupts Statistics

```python
import rasterio
import numpy as np

# ❌ nodata = -9999, but you compute mean including it
with rasterio.open('dem.tif') as src:
    elev = src.read(1)
    print(elev.mean())   # → -847.3 (pulled down by -9999 pixels!)

# ✅ Three correct approaches:
with rasterio.open('dem.tif') as src:
    # Approach 1: masked=True (preferred)
    elev = src.read(1, masked=True)
    print(float(elev.mean()))            # Masked array ignores nodata automatically

    # Approach 2: explicit filter
    elev = src.read(1)
    valid = elev[elev != src.nodata]
    print(valid.mean())

    # Approach 3: np.nan replacement
    elev = src.read(1).astype(np.float32)
    elev[elev == src.nodata] = np.nan
    print(np.nanmean(elev))
```

### CRS Mismatch Between Raster and Vector

```python
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import mapping

# ❌ PROBLEM: Vector in EPSG:4326, raster in EPSG:32633 → clip fails or returns empty
gdf = gpd.read_file('roi.shp')   # EPSG:4326

with rasterio.open('utm_image.tif') as src:  # EPSG:32633
    # Clipping with mismatched CRS → geometry falls outside raster bounds
    out, _ = mask(src, [mapping(gdf.geometry[0])], crop=True)  # Empty or error!

# ✅ GOOD: Always reproject vector to match raster CRS before spatial ops
with rasterio.open('utm_image.tif') as src:
    gdf_aligned = gdf.to_crs(src.crs)    # Reproject vector → raster's CRS
    out, _ = mask(src, [mapping(gdf_aligned.geometry[0])], crop=True)  # Correct!
```

### Reading Huge Files Causes OOM

```python
import rasterio
import numpy as np

# ❌ PROBLEM: 10GB GeoTIFF, read() loads everything into RAM
with rasterio.open('huge_satellite.tif') as src:
    data = src.read()   # MemoryError on 16GB machine

# ✅ GOOD: Read in blocks using block_windows()
with rasterio.open('huge_satellite.tif') as src:
    for ji, window in src.block_windows(1):
        block = src.read(window=window)   # Only this tile in memory
        # Process block...
        # Write result with same window...

# ✅ ALSO GOOD: Read at reduced resolution (overview)
with rasterio.open('huge_satellite.tif') as src:
    # Read at 1/8 native resolution — fits in memory, good for exploration
    small = src.read(out_shape=(src.count, src.height // 8, src.width // 8))
```

---

Rasterio's core value is the bridge between **geographic coordinates** and **pixel arrays**. Every operation — read, clip, reproject, write — maintains that bridge through the transform and CRS metadata. Master the read-compute-write pattern with proper metadata propagation, and you can build any geospatial processing pipeline from satellite data down to final maps.