---
name: astropy
description: The core library for Astronomy and Astrophysics in Python. Provides data structures for coordinates, time, units, FITS files, and cosmological models. Essential for observational data reduction and theoretical astrophysics. Use when working with astronomical coordinates (RA/Dec), physical units, FITS files, time scales, WCS, cosmology, or astronomical tables.
version: 6.1
license: BSD-3-Clause
---

# Astropy - Astronomy & Astrophysics

Astropy is more than a library; it's a community standard. It ensures that astronomical calculations are reproducible and physically consistent by linking numerical arrays with physical units and celestial reference frames.

## When to Use

- Handling physical units and constants in calculations.
- Working with celestial coordinates (Right Ascension, Declination) and frame transformations.
- Reading/writing FITS (Flexible Image Transport System), ASDF, and VO (Virtual Observatory) files.
- Managing astronomical time scales (UTC, TAI, TDB, Julian Dates).
- World Coordinate System (WCS) mapping (pixels to sky coordinates).
- Cosmological calculations (distances, ages, Hubble parameters).
- Tabular data with attached units and metadata (astropy.table).

## Reference Documentation

**Official docs**: https://docs.astropy.org/  
**Learn Astropy**: https://learn.astropy.org/  
**Search patterns**: `astropy.units.Quantity`, `astropy.coordinates.SkyCoord`, `astropy.io.fits`, `astropy.time.Time`

## Core Principles

### Units and Quantities
Everything in Astropy should be a Quantity — a number or array paired with a Unit. This prevents errors like adding meters to feet.

### Celestial Coordinates
Coordinates are represented by SkyCoord objects, which handle the complex math of spherical trigonometry and frame precession automatically.

### Tables with Units
While Pandas is great for dataframes, astropy.table.QTable is superior for physics because it preserves units during operations.

## Quick Reference

### Installation

```bash
pip install astropy
```

### Standard Imports

```python
import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import QTable
from astropy.io import fits
from astropy.wcs import WCS
```

### Basic Pattern - Physical Calculation

```python
# Calculate Schwarzschild radius of the Sun
mass_sun = 1.0 * u.M_sun
rs = (2 * const.G * mass_sun) / (const.c**2)

print(f"Radius: {rs.to(u.km):.2f}")
# Output: Radius: 2.95 km
```

## Critical Rules

### ✅ DO

- **Use astropy.units** - Always attach units to raw numbers from the start.
- **Prefer QTable** - Use QTable instead of Table to keep columns as Quantity objects.
- **Use SkyCoord for logic** - Never manually calculate angular separations; use `c1.separation(c2)`.
- **Vectorize coordinates** - SkyCoord can hold arrays of positions, which is much faster than loops.
- **Specify Time Scales** - Always define the scale (utc, tdb) when creating Time objects.
- **Use with for FITS** - Always use context managers to ensure file handles are closed.

### ❌ DON'T

- **Strip units early** - Avoid using `.value` until you absolutely need to pass data to a non-astropy function.
- **Hardcode constants** - Don't use 3e8 for speed of light; use `const.c`.
- **Assume J2000** - Be explicit about the frame (ICRS, FK5, Galactic).
- **Ignore WCS** - Don't assume pixels are linear; use the WCS object for mapping.

## Anti-Patterns (NEVER)

```python
# ❌ BAD: Manual unit conversion (Error prone)
distance_pc = 10.0
distance_m = distance_pc * 3.086e16 

# ✅ GOOD: Automatic conversion
dist = 10.0 * u.pc
dist_m = dist.to(u.m)

# ❌ BAD: Stripping units to use with NumPy functions
# (Astropy Quantities are NumPy-compatible, no need to strip!)
import numpy as np
vals = [1, 2, 3] * u.m
res = np.mean(vals.value) # ❌ Unit info lost

# ✅ GOOD: Keep units
res = np.mean(vals) # ✅ Returns <Quantity 2. m>

# ❌ BAD: Manual RA/Dec math
# sep = np.sqrt((ra1-ra2)**2 + (dec1-dec2)**2) # ❌ Incorrect for spherical geometry

# ✅ GOOD: Proper separation
c1 = SkyCoord(ra=10*u.deg, dec=20*u.deg)
c2 = SkyCoord(ra=11*u.deg, dec=21*u.deg)
sep = c1.separation(c2)
```

## Units and Constants (astropy.units)

### Complex Conversions

```python
# Equivalencies (e.g., converting wavelength to frequency)
wav = 500 * u.nm
freq = wav.to(u.THz, equivalencies=u.spectral())

# Flux conversions
flux_density = 15 * u.Jy
obs_freq = 1.4 * u.GHz
brightness_temp = flux_density.to(u.K, equivalencies=u.brightness_temperature(obs_freq, beam_area=1*u.sr))

# Solar units
lum = 3.8e26 * u.W
print(lum.to(u.L_sun))
```

## Tables (astropy.table)

### Managing Astronomical Catalogs

```python
# Create a QTable (Quantity Table)
tbl = QTable()
tbl['source_id'] = ['Star A', 'Star B', 'Star C']
tbl['flux'] = [1.2, 2.5, 0.8] * u.mJy
tbl['distance'] = [10, 25, 100] * u.pc

# Filtering
nearby = tbl[tbl['distance'] < 50 * u.pc]

# New column based on existing ones
tbl['luminosity'] = 4 * np.pi * tbl['distance']**2 * tbl['flux']

# Exporting
tbl.write('catalog.fits', overwrite=True)
tbl.write('catalog.csv', format='ascii.csv')
```

## Coordinates (astropy.coordinates)

### Frame Transformations

```python
# Create coordinate from string
c = SkyCoord("05h35m17.3s", "-05d23m28s", frame='icrs')

# Convert to Galactic coordinates
gal = c.galactic
print(f"L: {gal.l:.2f}, B: {gal.b:.2f}")

# Transform to AltAz (Horizontal) for a specific location and time
from astropy.coordinates import EarthLocation, AltAz
loc = EarthLocation(lat=45*u.deg, lon=-110*u.deg, height=2000*u.m)
time = Time.now()

altaz = c.transform_to(AltAz(obstime=time, location=loc))
print(f"Altitude: {altaz.alt:.2f}")
```

## Time (astropy.time)

### Precise Timekeeping

```python
# Julian Date vs ISO
t = Time("2023-12-01T12:00:00", scale='utc')
print(t.jd)
print(t.mjd) # Modified Julian Date

# Time deltas
dt = 24 * u.hour
future_t = t + dt

# Siderial Time
print(t.sidereal_time('mean', longitude=-110*u.deg))
```

## FITS and WCS (astropy.io.fits, astropy.wcs)

### Image Data Handling

```python
# Reading FITS
with fits.open("galaxy.fits") as hdul:
    data = hdul[0].data
    header = hdul[0].header
    
    # Initialize WCS from header
    w = WCS(header)

# Convert Pixel to World (Sky)
pix_coords = [[100, 100], [200, 200]]
world_coords = w.pixel_to_world(pix_coords[0][0], pix_coords[0][1])

# Convert Sky to Pixel
target = SkyCoord(ra=150.0*u.deg, dec=2.0*u.deg, frame='icrs')
px, py = w.world_to_pixel(target)
```

## Practical Workflows

### 1. Distance Calculation from Parallax

```python
def get_absolute_magnitude(mag, parallax_mas):
    """Calculate absolute magnitude given apparent mag and parallax in mas."""
    dist = (parallax_mas * u.mas).to(u.pc, equivalencies=u.parallax())
    # Absolute mag M = m - 5*log10(d/10pc)
    abs_mag = mag - 5 * np.log10(dist / (10 * u.pc))
    return abs_mag

# Star with mag 15 and 2 mas parallax
M = get_absolute_magnitude(15, 2)
```

### 2. Aperture Photometry Units Correction

```python
def flux_to_mag(flux, zero_point=25.0):
    """Convert raw flux to magnitudes with unit checking."""
    if not isinstance(flux, u.Quantity):
        raise ValueError("Flux must have units (e.g., u.electron / u.s)")
    
    # Assuming zero_point is defined for 1 electron/s
    mag = -2.5 * np.log10(flux / (1 * u.electron / u.s)) + zero_point
    return mag
```

### 3. Cosmological Distance Evolution

```python
from astropy.cosmology import FlatLambdaCDM

# Define cosmology (Planck 2018 defaults)
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)

z = np.linspace(0, 5, 100)
dist = cosmo.luminosity_distance(z) # Returns distance in Mpc

print(f"Lookback time at z=1: {cosmo.lookback_time(1).to(u.Gyr):.2f}")
```

## Performance Optimization

### Using astropy.utils.console for Long Tasks

```python
from astropy.utils.console import ProgressBar

with ProgressBar(len(large_catalog)) as bar:
    for source in large_catalog:
        # Complex calculation
        bar.update()
```

### Fast I/O with C Engine

```python
# When reading huge CSV catalogs
tbl = QTable.read('massive_data.csv', format='ascii.fast_csv')
```

## Common Pitfalls and Solutions

### The "Unit Stripping" bug

```python
# ❌ Problem: Using a library that doesn't support Quantities
# solution = some_external_lib(data * u.m) # Crashes!

# ✅ Solution: Wrap the function or convert explicitly
data_to_pass = (data * u.m).to_value(u.cm) # Strip unit but ensure it's in CM
```

### SkyCoord performance in loops

```python
# ❌ Problem: Creating SkyCoord objects inside a loop is very slow
for r, d in zip(ra_list, dec_list):
    c = SkyCoord(r, d, unit='deg')

# ✅ Solution: Create an array of coordinates
c_array = SkyCoord(ra=ra_list*u.deg, dec=dec_list*u.deg)
```

### Decomposing units

```python
# ❌ Problem: Result has messy units like 'm / km'
res = (10 * u.m) / (2 * u.km)

# ✅ Solution: Use .decompose() or .simplify()
print(res.decompose()) # Result: 0.005 (dimensionless)
```

Astropy is the backbone of reproducible research in astronomy. By enforcing physical consistency through units and standardizing coordinate handling, it prevents the silent errors that used to plague astronomical software.
