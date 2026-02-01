---
name: photutils
description: An Astropy coordinated package for detecting and performing photometry of astronomical sources. Provides tools for background estimation, source detection (DAOFIND, IRAF), aperture photometry, and PSF (Point Spread Function) fitting. Use when working with astronomical image analysis, star/galaxy detection, measuring brightness (photometry), background subtraction, PSF fitting, aperture photometry, centroiding, or isophotal analysis.
version: 1.10
license: BSD-3-Clause
---

# Photutils - Astronomical Photometry

Photutils is the modern standard for extracting quantitative data from astronomical images. It replaces legacy tools like SExtractor or DAOPHOT with a modular, Pythonic interface integrated with the Astropy ecosystem.

## When to Use

- Detecting stars or galaxies in an astronomical image (DAOStarFinder, IRAFStarFinder).
- Measuring the brightness of objects using circular, elliptical, or rectangular apertures.
- Performing PSF-fitting photometry (essential for crowded stellar fields).
- Estimating and subtracting complex 2D backgrounds (sky background).
- Calculating the centroids (precise centers) of astronomical sources.
- Performing "isophotal analysis" (measuring shapes of galaxies).
- Segmentation of images based on thresholding.

## Reference Documentation

**Official docs**: https://photutils.readthedocs.io/  
**GitHub**: https://github.com/astropy/photutils  
**Search patterns**: `photutils.aperture`, `photutils.detection`, `photutils.background`, `photutils.psf`

## Core Principles

### Background First
Accurate photometry is impossible without correct background subtraction. Photutils provides tools to estimate global and local sky levels.

### Aperture vs. PSF
- **Aperture Photometry**: Summing pixels within a fixed shape. Best for isolated sources or galaxies.
- **PSF Photometry**: Fitting a mathematical model of a star to the data. Best for overlapping stars or high-precision work.

### Units and WCS
Photutils is fully units-aware. If your image has WCS metadata, apertures can be defined in sky coordinates (degrees) rather than just pixels.

## Quick Reference

### Installation

```bash
pip install photutils
```

### Standard Imports

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground
```

### Basic Pattern - Detection and Aperture Photometry

```python
import numpy as np
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry

# 1. Detect sources
# fwhm: Full Width at Half Maximum of the stars
finder = DAOStarFinder(fwhm=3.0, threshold=5.*std)
sources = finder(image - background)

# 2. Define apertures at detected positions
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.0)

# 3. Perform photometry
phot_table = aperture_photometry(image - background, apertures)
print(phot_table)
```

## Critical Rules

### ✅ DO

- **Subtract background first** - Detection and photometry always work better on a background-subtracted image.
- **Use Background2D** - For images with varying sky levels, a 2D mesh estimation is more accurate than a single median value.
- **Specify the Gain** - For accurate error (uncertainty) estimation, provide the detector gain to `aperture_photometry`.
- **Match FWHM** - Ensure the `fwhm` parameter in finders matches the actual PSF of your images.
- **Vectorize apertures** - One `CircularAperture` object can hold thousands of positions; don't create them in a loop.
- **Use SkyAperture** - If you have a WCS, define apertures in RA/Dec to ensure they stay on target if the image is rotated or reprojected.

### ❌ DON'T

- **Ignore mask pixels** - If your image has bad pixels (cosmic rays, saturated stars), pass a mask to the photometry functions.
- **Use small apertures** - If an aperture is too small, you lose light (aperture correction required). If too large, you add noise.
- **Blindly trust defaults** - Always visualize your apertures on top of the image to verify they align with the stars.
- **Mix pixel and sky units** - Be careful whether you are passing (x, y) or (RA, Dec).

## Anti-Patterns (NEVER)

```python
# ❌ BAD: Manual pixel summation
# light_sum = np.sum(image[y-r:y+r, x-r:x+r]) # No circular masking!

# ✅ GOOD: Proper aperture photometry
from photutils.aperture import CircularAperture
ap = CircularAperture((x, y), r=r)
phot = aperture_photometry(image, ap)

# ❌ BAD: Detecting sources on raw data with high background
# finder = DAOStarFinder(threshold=100)(raw_image)

# ✅ GOOD: Detection on noise-subtracted data
from photutils.background import Background2D, MedianBackground
bkg = Background2D(image, (50, 50), filter_size=(3, 3), bkg_estimator=MedianBackground())
sources = finder(image - bkg.background)
```

## Background Estimation (photutils.background)

### Robust 2D Background Mapping

```python
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip

sigma_clip = SigmaClip(sigma=3.0)
bkg_estimator = MedianBackground()

# box_size: size of the mesh grid
bkg = Background2D(image, box_size=(50, 50), 
                   filter_size=(3, 3), 
                   sigma_clip=sigma_clip, 
                   bkg_estimator=bkg_estimator)

print(f"Global median background: {bkg.background_median}")
clean_image = image - bkg.background
```

## Source Detection (photutils.detection)

### Star Finding Algorithms

```python
from photutils.detection import DAOStarFinder, IRAFStarFinder

# DAOStarFinder: Optimized for stars
daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)
sources = daofind(data)

# IRAFStarFinder: Alternative implementation
iraffind = IRAFStarFinder(threshold=5.0*std, fwhm=3.0)
sources_iraf = iraffind(data)

# Extract coordinates for photometry
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
```

## Aperture Photometry (photutils.aperture)

### Handling Multiple Shapes

```python
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# 1. Main aperture and Annulus (for local background)
positions = [(100.5, 120.3), (250.1, 88.4)]
aperture = CircularAperture(positions, r=5.0)
annulus = CircularAnnulus(positions, r_in=10.0, r_out=15.0)

# 2. Local background subtraction
# Calculate median in the annulus
annulus_masks = annulus.to_mask(method='center')
bkg_median = []
for mask in annulus_masks:
    annulus_data = mask.multiply(image)
    annulus_data_1d = annulus_data[mask.data > 0]
    bkg_median.append(np.median(annulus_data_1d))

# 3. Perform photometry and subtract local background
phot = aperture_photometry(image, aperture)
phot['local_bkg'] = bkg_median
phot['aper_sum_corrected'] = phot['aperture_sum'] - (bkg_median * aperture.area)
```

## PSF Photometry (photutils.psf)

### Fitting Overlapping Sources

```python
from photutils.psf import BasicPSFFitter, IntegratedGaussianPRF
from photutils.background import MMMBackground

# 1. Define the PSF model (Gaussian or custom)
psf_model = IntegratedGaussianPRF(sigma=1.5)

# 2. Initialize the fitter
fitter = BasicPSFFitter(psf_model, fitshape=(11, 11), 
                        finder=daofind, 
                        bkg_estimator=MMMBackground())

# 3. Perform fitting
# result_table contains fitted x, y, and flux
result_table = fitter(image)

# 4. Get the residual image (original - fitted stars)
residual_image = fitter.get_residual_image()
```

## Centroiding (photutils.centroids)

### Precise Star Centering

```python
from photutils.centroids import centroid_com, centroid_2dg

# Center of mass (simple)
x1, y1 = centroid_com(image[y-5:y+6, x-5:x+6])

# 2D Gaussian fit (most precise for stars)
x2, y2 = centroid_2dg(image[y-5:y+6, x-5:x+6])
```

## Practical Workflows

### 1. Full Image Pipeline (Detection to Magnitude)

```python
def extract_magnitudes(image, zero_point=25.0):
    # 1. Subtract 2D Background
    bkg = Background2D(image, box_size=(64, 64))
    data_sub = image - bkg.background
    
    # 2. Detect sources
    finder = DAOStarFinder(fwhm=2.5, threshold=5*bkg.background_rms_median)
    sources = finder(data_sub)
    
    # 3. Aperture Photometry
    pos = np.transpose((sources['xcentroid'], sources['ycentroid']))
    ap = CircularAperture(pos, r=4.0)
    phot = aperture_photometry(data_sub, ap)
    
    # 4. Convert to instrumental magnitudes
    phot['mag'] = -2.5 * np.log10(phot['aperture_sum']) + zero_point
    return phot
```

### 2. Checking Photometry Quality with Residuals

```python
def check_psf_quality(image, sources):
    """Fits PSF and returns residual to check for artifacts."""
    # ... setup PSF fitter ...
    result = fitter(image, init_guesses=sources)
    residual = fitter.get_residual_image()
    
    # If residual has 'holes' or 'donuts', PSF model is bad.
    return residual
```

## Performance Optimization

### Using n_jobs in fitting
While current Photutils doesn't natively support `n_jobs` in all fitters, you can parallelize by splitting the image into tiles.

### Aperture Masks
For very large images, use `aperture.to_mask()` to extract only relevant sub-arrays instead of performing calculations on the full image.

## Common Pitfalls and Solutions

### Negative Flux in log10
If the background subtraction is too aggressive, some faint stars might have negative `aperture_sum`.

```python
# ✅ Solution: Filter results before log
valid = phot[phot['aperture_sum'] > 0]
valid['mag'] = -2.5 * np.log10(valid['aperture_sum'])
```

### Crowded Fields
Aperture photometry fails when stars overlap.

```python
# ❌ Problem: One aperture contains 3 stars
# ✅ Solution: Switch to PSF Photometry (photutils.psf)
# This fits the shapes and accounts for overlapping light.
```

### Coordinate Order
photutils uses (x, y) which corresponds to (column, row).

```python
# ❌ Error: image[x, y] # This is wrong for numpy!
# ✅ Solution: Use image[y, x] or image[row, col]
```

Photutils is the "measuring stick" for astronomical imagery. By combining robust statistical detection with precise mathematical fitting, it allows astronomers to turn digital images into scientific catalogs with known uncertainties.
