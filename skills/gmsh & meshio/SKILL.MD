---
name: gmsh-meshio
description: Programmatic mesh generation and mesh I/O for computational physics and FEM simulation. gmsh generates 2D and 3D meshes from geometric primitives and CAD-style boolean operations (union, difference, intersection) via the OpenCASCADE kernel, with fine-grained control over element size, adaptive refinement around features, and physical group tagging for boundary conditions. meshio reads and writes meshes across 40+ formats (GeoTIFF, VTK, VTU, Gmsh .msh, HDF5, ExodusII, XDMF, NetCDF) and performs format conversion. Use when: generating meshes for FEM/FEA simulation (prerequisite for FEniCS, deal.II, Firedrake), creating 2D or 3D computational domains from geometric descriptions, performing CSG (Constructive Solid Geometry) to build complex shapes from primitives, controlling mesh density and adaptive refinement near boundaries or singularities, tagging boundaries and subdomains for boundary conditions in solvers, converting meshes between simulation formats, or inspecting and manipulating mesh data programmatically. This is the geometry layer that sits between "I have a shape" and "I can simulate physics on it."
version: gmsh 4.12 / meshio 5.3
license: GPL-2.0 (gmsh) / MIT (meshio)
---

# gmsh + meshio — Mesh Generation and I/O

Two libraries, one pipeline: **gmsh** builds and meshes geometry, **meshio** moves meshes between formats. Together they are the standard way to go from "I need a computational domain" to "I have a mesh file my solver can read."

## Core Mental Model

```
WHAT IS A MESH?
  A geometric domain discretized into elements.
  
  2D: triangles (tri3)    or quadrilaterals (quad4)
  3D: tetrahedra (tet4)   or hexahedra (hex8)
  
  A mesh has:
    • Nodes (points)         — coordinates in physical space
    • Elements (cells)       — connectivity: which nodes form each element
    • Boundary tags          — which elements/edges are on which boundary
    • Physical groups        — named collections of entities (for BCs in FEM)

THE PIPELINE:
  
  Geometry (points, curves, surfaces, volumes)
       ↓  gmsh kernel: OCC (OpenCASCADE) or built-in
  Mesh generation (gmsh.model.mesh.generate)
       ↓
  .msh file  ←→  meshio  ←→  .vtk / .vtu / .xdmf / .h5 / ...
       ↓
  FEM solver (FEniCS, deal.II, Firedrake)

TWO GMSH KERNELS:
  • OCC (OpenCASCADE)  — USE THIS. Full CSG boolean ops, robust for complex geometry.
  • built-in           — Simpler, faster for trivial shapes, but limited boolean support.
  Set kernel: gmsh.option.setNumber("Geometry.OCCImportParametricCAD", 1)  (default in modern gmsh)
```

## Reference Documentation

**gmsh docs**: https://gmsh.info/doc/texinfo/gmsh.html  
**gmsh Python API**: https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes  
**meshio GitHub**: https://github.com/sigma-py/meshio  
**meshio docs**: https://meshio.readthedocs.io/  
**Search patterns**: `gmsh.model.occ`, `gmsh.model.mesh.generate`, `meshio.read`, `physical groups`

## Core Principles

### gmsh Must Be Initialized and Finalized
Every gmsh session starts with `gmsh.initialize()` and ends with `gmsh.finalize()`. All geometry and mesh operations happen between these two calls. Forgetting `finalize()` leaves resources dangling. Use a try/finally block or context pattern.

### OCC Kernel: Build Then Mesh
With the OCC kernel, you first build the full geometry (all points, curves, surfaces, volumes, boolean operations), then call `gmsh.model.occ.synchronize()` to push geometry into the gmsh model, THEN mesh. The synchronize step is mandatory — without it, the mesher sees nothing.

### Physical Groups Are the Bridge to FEM
Physical groups assign names/tags to geometric entities (surfaces, volumes, curves). FEM solvers use these tags to know where to apply boundary conditions. Defining physical groups correctly is the single most important step for FEM workflows. If you skip them, your solver has no way to distinguish boundaries.

### Mesh Size = Element Size at Nodes
`gmsh.model.mesh.SetSizeAtPoints()` or size fields control how fine the mesh is. Smaller size = more elements = more accurate but slower to solve. Adaptive size fields let you refine only where it matters (boundaries, singularities, regions of interest).

### meshio Is Format-Agnostic
meshio doesn't care what format you give it. `meshio.read('mesh.msh')` and `meshio.read('mesh.vtk')` return the same `Mesh` object. Write to any supported format with `mesh.write('output.vtu')`.

## Quick Reference

### Installation

```bash
pip install gmsh meshio numpy matplotlib
# For visualization:
pip install pyvista   # 3D interactive visualization
# For FEniCS integration:
# pip install fenics-dolfinx   # See FEniCS docs for install
```

### Standard Imports

```python
import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
```

### Basic Pattern — 2D Rectangle Mesh

```python
import gmsh
import meshio

gmsh.initialize()

# Use OCC kernel (required for boolean operations)
gmsh.model.occ.addRectangle(x=0, y=0, z=0, dx=2.0, dy=1.0)  # tag=1 auto-assigned

# Synchronize geometry → mesher
gmsh.model.occ.synchronize()

# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Save
gmsh.write('rectangle.msh')
gmsh.finalize()

# Read with meshio
mesh = meshio.read('rectangle.msh')
print(f"Points: {mesh.points.shape}")          # (N, 3)
print(f"Triangles: {mesh.cells_dict['triangle'].shape}")  # (M, 3)
```

### Basic Pattern — 3D Box Mesh

```python
import gmsh

gmsh.initialize()

gmsh.model.occ.addBox(x=0, y=0, z=0, dx=1.0, dy=1.0, dz=1.0)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write('box.msh')

gmsh.finalize()
```

## Critical Rules

### ✅ DO

- **Always call `gmsh.model.occ.synchronize()` after geometry operations, before meshing** — Without this, the mesher sees an empty model. This is the #1 source of "mesh is empty" bugs.
- **Use the OCC kernel (`gmsh.model.occ`) for anything beyond trivial geometry** — It's the only kernel with full boolean operations. Use `gmsh.model.geo` only for simple extrusions where you need fine control over entity tags.
- **Define physical groups BEFORE writing the mesh** — Physical groups tag the entities. They must exist at write time or they won't be in the file.
- **Use try/finally to guarantee `gmsh.finalize()`** — If mesh generation crashes, finalize still runs. Resource leak otherwise.
- **Control mesh size explicitly** — Don't rely on gmsh defaults. Set `gmsh.option.setNumber("Mesh.MeshSizeMax", ...)` globally or use size fields locally.
- **Use `gmsh.model.mesh.SetSizeAtPoints()` for local refinement** — Finer mesh near boundaries, coarser in bulk. Standard practice in FEM.
- **Verify the mesh after generation** — Check node/element counts, check that physical groups contain the expected entities. Silent failures are common.
- **Use `meshio` for any format conversion** — Don't hand-roll mesh parsers.

### ❌ DON'T

- **Don't mix OCC and built-in kernel calls** — `gmsh.model.occ.*` and `gmsh.model.geo.*` are two separate kernels. Mixing them produces undefined behavior.
- **Don't forget `synchronize()` after boolean operations** — Boolean ops modify geometry in the OCC kernel's internal state. `synchronize()` pushes that state to gmsh. Without it: nothing changed.
- **Don't assume entity tags are stable** — Boolean operations (union, difference) destroy input entities and create new ones with new tags. Always retrieve tags AFTER the boolean op.
- **Don't define physical groups using tags from before a boolean operation** — Those tags no longer exist. Get the output tags from the boolean op return value.
- **Don't mesh in 3D if you only need 2D** — `generate(3)` on a 2D geometry wastes time. Match dimension to your problem.
- **Don't ignore the `dimTag` convention** — gmsh uses `(dim, tag)` pairs everywhere. dim=0 is point, 1 is curve, 2 is surface, 3 is volume. Always track both.

## Anti-Patterns (NEVER)

```python
import gmsh

# ❌ BAD: No synchronize before meshing
gmsh.initialize()
gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
# gmsh.model.occ.synchronize()   ← MISSING
gmsh.model.mesh.generate(3)      # Meshes NOTHING — 0 elements
gmsh.write('empty.msh')
gmsh.finalize()

# ✅ GOOD: Synchronize before mesh
gmsh.initialize()
gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
gmsh.model.occ.synchronize()     # ← Required
gmsh.model.mesh.generate(3)      # Now meshes the box correctly
gmsh.write('box.msh')
gmsh.finalize()

# ─────────────────────────────────────────────────

# ❌ BAD: Physical groups with stale tags after boolean ops
gmsh.initialize()
box1 = gmsh.model.occ.addBox(0,0,0, 2,2,2)     # tag=1
box2 = gmsh.model.occ.addBox(0.5,0.5,0.5, 1,1,1)  # tag=2
gmsh.model.occ.booleanDifference([(3,box1)], [(3,box2)])  # box1 is DESTROYED, new entity created
gmsh.model.occ.synchronize()
gmsh.model.mesh.addPhysicalGroup(3, [box1])      # ← WRONG: box1 tag no longer exists!

# ✅ GOOD: Capture output tags from boolean op
gmsh.initialize()
box1 = gmsh.model.occ.addBox(0,0,0, 2,2,2)
box2 = gmsh.model.occ.addBox(0.5,0.5,0.5, 1,1,1)
out, out_dim_tags = gmsh.model.occ.booleanDifference([(3,box1)], [(3,box2)])
# out = [(3, new_tag)] — the resulting volume
result_tag = out[0][1]
gmsh.model.occ.synchronize()
gmsh.model.mesh.addPhysicalGroup(3, [result_tag], name="solid")   # ← Correct tag

# ─────────────────────────────────────────────────

# ❌ BAD: No try/finally — gmsh.finalize() never called on error
gmsh.initialize()
gmsh.model.occ.addBox(0,0,0,1,1,1)
raise ValueError("something broke")   # finalize() never runs → resource leak
gmsh.finalize()

# ✅ GOOD: Protected finalize
gmsh.initialize()
try:
    gmsh.model.occ.addBox(0,0,0,1,1,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write('box.msh')
finally:
    gmsh.finalize()   # Always runs
```

## gmsh — Geometry Building Blocks

### Primitives (OCC Kernel)

```python
import gmsh

gmsh.initialize()

# ─── 2D primitives ───
rect  = gmsh.model.occ.addRectangle(x=0, y=0, z=0, dx=4.0, dy=2.0)   # Returns tag
disk  = gmsh.model.occ.addDisk(cx=2.0, cy=1.0, z=0, dx=1.0, dy=1.0)  # Ellipse (dx=dy → circle)

# ─── 3D primitives ───
box      = gmsh.model.occ.addBox(x=0, y=0, z=0, dx=2, dy=2, dz=2)
sphere   = gmsh.model.occ.addSphere(xc=0, yc=0, zc=0, radius=1.0)
cylinder = gmsh.model.occ.addCylinder(x=0, y=0, z=0, dx=0, dy=0, dz=2, radius=0.5)
cone     = gmsh.model.occ.addCone(x=0, y=0, z=0, dx=0, dy=0, dz=2, r1=1.0, r2=0.3)
torus    = gmsh.model.occ.addTorus(x=0, y=0, z=0, majorRadius=2, minorRadius=0.5)

# ─── Extrusion: 2D shape → 3D volume ───
# First create a 2D profile, then extrude it along a vector
rect_tag = gmsh.model.occ.addRectangle(0, 0, 0, 2, 1)
gmsh.model.occ.synchronize()

# Extrude surface (dim=2) along z-axis by 3.0 units
# Returns list of (dim, tag) for all generated entities
extruded = gmsh.model.occ.extrude(
    dimTags=[(2, rect_tag)],
    dx=0, dy=0, dz=3.0
)
# extruded contains the new volume, lateral surfaces, top surface

gmsh.model.occ.synchronize()
gmsh.finalize()
```

### Boolean Operations (CSG)

```python
import gmsh

gmsh.initialize()

# ─── UNION: combine two volumes into one ───
box1 = gmsh.model.occ.addBox(0, 0, 0, 2, 2, 2)
box2 = gmsh.model.occ.addBox(1, 1, 0, 2, 2, 2)   # Overlapping

# booleanUnion(objectDimTags, toolDimTags)
# Object = base shape. Tool = shape to unite with.
out, _ = gmsh.model.occ.booleanUnion(
    objectDimTags=[(3, box1)],
    toolDimTags=[(3, box2)]
)
# out = [(3, result_tag)] — single unified volume
gmsh.model.occ.synchronize()
gmsh.finalize()

# ─── DIFFERENCE: subtract tool from object ───
gmsh.initialize()
outer = gmsh.model.occ.addBox(0, 0, 0, 4, 4, 4)
hole  = gmsh.model.occ.addCylinder(2, 2, -1, 0, 0, 6, radius=1.0)  # Through-hole along z

out, _ = gmsh.model.occ.booleanDifference(
    objectDimTags=[(3, outer)],
    toolDimTags=[(3, hole)]
)
# Result: box with a cylindrical hole
gmsh.model.occ.synchronize()
gmsh.finalize()

# ─── INTERSECTION: keep only the overlapping region ───
gmsh.initialize()
sphere = gmsh.model.occ.addSphere(0, 0, 0, 2.0)
box    = gmsh.model.occ.addBox(-1, -1, -1, 2, 2, 2)

out, _ = gmsh.model.occ.booleanIntersection(
    objectDimTags=[(3, sphere)],
    toolDimTags=[(3, box)]
)
# Result: the part of the sphere inside the box
gmsh.model.occ.synchronize()
gmsh.finalize()

# ─── FRAGMENT: split geometries at intersections (keeps all pieces) ───
gmsh.initialize()
box1 = gmsh.model.occ.addBox(0, 0, 0, 2, 2, 2)
box2 = gmsh.model.occ.addBox(1, 0, 0, 2, 2, 2)   # Overlapping

# Fragment splits both into non-overlapping pieces
out, out_dim_tags = gmsh.model.occ.fragment(
    objectDimTags=[(3, box1)],
    toolDimTags=[(3, box2)]
)
# out contains ALL resulting volumes (3 pieces: left-only, overlap, right-only)
gmsh.model.occ.synchronize()
gmsh.finalize()
```

### Mesh Size Control

```python
import gmsh

gmsh.initialize()

# ─── Global mesh size ───
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)   # Largest element edge length
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)  # Smallest element edge length

# ─── Per-point size (most common approach) ───
# Create geometry with explicit size at each point
p1 = gmsh.model.occ.addPoint(0, 0, 0, meshSize=0.1)   # Fine near origin
p2 = gmsh.model.occ.addPoint(1, 0, 0, meshSize=0.5)   # Coarse at corners
p3 = gmsh.model.occ.addPoint(1, 1, 0, meshSize=0.5)
p4 = gmsh.model.occ.addPoint(0, 1, 0, meshSize=0.1)

# Build curves and surface from these points
c1 = gmsh.model.occ.addLine(p1, p2)
c2 = gmsh.model.occ.addLine(p2, p3)
c3 = gmsh.model.occ.addLine(p3, p4)
c4 = gmsh.model.occ.addLine(p4, p1)
cl = gmsh.model.occ.addCurveLoop([c1, c2, c3, c4])
s  = gmsh.model.occ.addPlaneSurface([cl])

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write('adaptive.msh')
gmsh.finalize()

# ─── Size fields (most powerful: spatial functions) ───
gmsh.initialize()
gmsh.model.occ.addRectangle(0, 0, 0, 4, 4)
gmsh.model.occ.synchronize()

# Field 1: Distance from a point (measures distance to geometry)
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "PointsList", [1])   # Point tag 1

# Field 2: Threshold — map distance to element size
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)         # Uses Field 1 (distance)
gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.05)      # Size near the point
gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.5)       # Size far away
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)       # Distance where SizeMin applies
gmsh.model.mesh.field.setNumber(2, "DistMax", 2.0)       # Distance where SizeMax applies

# Set as the background mesh size field
gmsh.model.mesh.field.setAsBackground(2)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)      # Disable default point-based sizing

gmsh.model.mesh.generate(2)
gmsh.write('refined_near_point.msh')
gmsh.finalize()
```

## Physical Groups — The FEM Bridge

Physical groups are how gmsh communicates boundary conditions to FEM solvers. Without them, FEniCS/deal.II cannot distinguish "this surface is a wall" from "this surface is an inlet."

```python
import gmsh

gmsh.initialize()

# Build a 2D domain: rectangle with a circular hole
rect = gmsh.model.occ.addRectangle(0, 0, 0, 4, 2)
disk = gmsh.model.occ.addDisk(2, 1, 0, 0.5, 0.5)

# Cut the hole
out, _ = gmsh.model.occ.booleanDifference(
    objectDimTags=[(2, rect)],
    toolDimTags=[(2, disk)]
)
result_surface = out[0][1]   # The surface with the hole

gmsh.model.occ.synchronize()

# ─── Get boundary curves ───
# After boolean op, boundary curves have new tags. Retrieve them.
boundaries = gmsh.model.occ.getBoundary([(2, result_surface)], oriented=False)
# boundaries = list of (dim=1, tag) for each boundary curve

# We need to identify WHICH curve is which.
# Strategy: check the center of mass of each curve.
inlet_curves   = []
outlet_curves  = []
wall_curves    = []
hole_curves    = []

for dim, tag in boundaries:
    com = gmsh.model.occ.getCenterOfMass(dim, tag)
    x, y, z = com

    if abs(x - 0.0) < 1e-6:         # Left edge → inlet
        inlet_curves.append(tag)
    elif abs(x - 4.0) < 1e-6:       # Right edge → outlet
        outlet_curves.append(tag)
    elif abs((x-2)**2 + (y-1)**2 - 0.25) < 0.1:  # Near circle → hole boundary
        hole_curves.append(tag)
    else:                            # Top/bottom → walls
        wall_curves.append(tag)

# ─── Define physical groups ───
# dim=1 for curves (boundaries in 2D), dim=2 for surfaces (domain)
gmsh.model.mesh.addPhysicalGroup(1, inlet_curves,   name="inlet")
gmsh.model.mesh.addPhysicalGroup(1, outlet_curves,  name="outlet")
gmsh.model.mesh.addPhysicalGroup(1, wall_curves,    name="walls")
gmsh.model.mesh.addPhysicalGroup(1, hole_curves,    name="hole")
gmsh.model.mesh.addPhysicalGroup(2, [result_surface], name="domain")

# ─── Mesh and save ───
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.15)
gmsh.model.mesh.generate(2)
gmsh.write('domain_with_hole.msh')

gmsh.finalize()

# ─── Verify with meshio ───
import meshio
mesh = meshio.read('domain_with_hole.msh')
print("Physical groups found:", list(mesh.point_sets.keys()) if mesh.point_sets else "none")
print("Cell sets:", list(mesh.cell_sets_dict.keys()))
# Should show: inlet, outlet, walls, hole, domain
```

## meshio — Mesh I/O and Inspection

### Read, Inspect, Write

```python
import meshio
import numpy as np

# ─── Read any supported format ───
mesh = meshio.read('mesh.msh')     # gmsh
# mesh = meshio.read('mesh.vtk')   # VTK legacy
# mesh = meshio.read('mesh.vtu')   # VTK XML
# mesh = meshio.read('mesh.h5')    # HDF5

# ─── Inspect ───
print(f"Points (nodes): {mesh.points.shape}")       # (num_nodes, 3) — always 3D coords
print(f"Cell types: {list(mesh.cells_dict.keys())}")  # e.g., ['triangle', 'line', 'vertex']

for cell_type, cells in mesh.cells_dict.items():
    print(f"  {cell_type:15s}: {cells.shape[0]:>8,} elements, {cells.shape[1]} nodes/element")

# Point sets (named node groups) and cell sets (named element groups)
if mesh.point_sets:
    for name, indices in mesh.point_sets.items():
        print(f"  Point set '{name}': {len(indices)} nodes")
if mesh.cell_sets_dict:
    for name, indices in mesh.cell_sets_dict.items():
        total = sum(len(i) for i in indices)
        print(f"  Cell set  '{name}': {total} elements")

# ─── Write to any format ───
mesh.write('output.vtk')           # VTK legacy — compatible with ParaView
mesh.write('output.vtu')           # VTK XML — compressed, parallel-friendly
mesh.write('output.xdmf')          # XDMF — standard for FEniCS
mesh.write('output.exodus.e')      # ExodusII — standard for Abaqus/ANSYS

# ─── Format conversion (one-liner) ───
meshio.convert('input.msh', 'output.vtu')
```

### Extract Mesh Components

```python
import meshio
import numpy as np

mesh = meshio.read('domain.msh')

# ─── Node coordinates ───
points = mesh.points   # (N, 3) — x, y, z for every node
x, y, z = points[:, 0], points[:, 1], points[:, 2]

# ─── Element connectivity ───
# cells_dict: {cell_type: array of shape (num_cells, nodes_per_cell)}
triangles = mesh.cells_dict.get('triangle', np.array([]))   # 2D elements
tets      = mesh.cells_dict.get('tetra', np.array([]))      # 3D elements
lines     = mesh.cells_dict.get('line', np.array([]))       # Boundary edges (2D)
quads     = mesh.cells_dict.get('quad', np.array([]))       # Quad elements

# ─── Access physical group (boundary) node indices ───
# Physical groups on curves become "line" elements tagged in cell_sets
if 'inlet' in mesh.cell_sets_dict:
    # cell_sets_dict returns a list of arrays (one per cell type block)
    inlet_cell_indices = mesh.cell_sets_dict['inlet']
    # Get the line elements on the inlet
    for block_idx, cell_indices in enumerate(inlet_cell_indices):
        if len(cell_indices) > 0:
            cell_block = mesh.cells[block_idx]
            inlet_elements = cell_block.data[cell_indices]
            inlet_nodes    = np.unique(inlet_elements)
            print(f"Inlet: {len(inlet_nodes)} nodes, {len(cell_indices)} edges")

# ─── Compute mesh statistics ───
def mesh_stats(mesh):
    """Print summary statistics of a mesh."""
    print(f"Nodes:      {mesh.points.shape[0]:>10,}")
    for ctype, cells in mesh.cells_dict.items():
        print(f"{ctype:12s}: {cells.shape[0]:>10,} elements")
    # Bounding box
    mins = mesh.points.min(axis=0)
    maxs = mesh.points.max(axis=0)
    print(f"Bounds:     x=[{mins[0]:.3f}, {maxs[0]:.3f}]  "
          f"y=[{mins[1]:.3f}, {maxs[1]:.3f}]  "
          f"z=[{mins[2]:.3f}, {maxs[2]:.3f}]")

mesh_stats(mesh)
```

### Mesh Manipulation

```python
import meshio
import numpy as np

mesh = meshio.read('input.msh')

# ─── Add point data (e.g., initial condition for FEM) ───
# Initial temperature field: T = x^2 + y^2
T_init = mesh.points[:, 0]**2 + mesh.points[:, 1]**2
mesh.point_data['temperature'] = T_init

# ─── Add cell data (e.g., material ID per element) ───
n_tris = mesh.cells_dict['triangle'].shape[0]
material_id = np.zeros(n_tris, dtype=int)
# Assign material based on centroid position
centroids = mesh.points[mesh.cells_dict['triangle']].mean(axis=1)  # (N, 3)
material_id[centroids[:, 0] > 2.0] = 1   # Region 1 where x > 2
mesh.cell_data['material'] = [material_id]   # List wrapping required

# ─── Remove specific cell types (e.g., strip boundary elements) ───
# Keep only volumetric elements
keep_types = {'triangle', 'tetra', 'quad', 'hexahedron'}
mesh.cells = [block for block in mesh.cells if block.type in keep_types]

# ─── Write enriched mesh ───
mesh.write('enriched.vtu')
```

## Visualization

```python
import meshio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def plot_2d_mesh(mesh_path: str, color_field: str = None, title: str = 'Mesh'):
    """Plot a 2D triangular mesh, optionally colored by a point field."""
    mesh = meshio.read(mesh_path)
    points = mesh.points
    tris   = mesh.cells_dict.get('triangle', None)

    if tris is None:
        raise ValueError("No triangles found in mesh")

    # matplotlib Triangulation
    tri = Triangulation(points[:, 0], points[:, 1], tris)

    fig, ax = plt.subplots(figsize=(10, 6))

    if color_field and color_field in mesh.point_data:
        values = mesh.point_data[color_field]
        mappable = ax.tripcolor(tri, values, cmap='viridis', shading='gouraud')
        plt.colorbar(mappable, ax=ax, label=color_field)
    else:
        ax.triplot(tri, 'k-', linewidth=0.3, alpha=0.6)

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.show()

# plot_2d_mesh('domain.msh')
# plot_2d_mesh('solution.vtu', color_field='temperature', title='Temperature Field')


def plot_mesh_and_boundaries(mesh_path: str):
    """Plot 2D mesh with boundary physical groups highlighted."""
    mesh    = meshio.read(mesh_path)
    points  = mesh.points
    tris    = mesh.cells_dict.get('triangle')

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all triangles (light gray)
    if tris is not None:
        tri = Triangulation(points[:, 0], points[:, 1], tris)
        ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.4)

    # Overlay boundary edges colored by physical group
    colors = {'inlet': 'blue', 'outlet': 'red', 'walls': 'green',
              'hole': 'orange', 'symmetry': 'purple'}

    for name, color in colors.items():
        if name in mesh.cell_sets_dict:
            for block_idx, cell_indices in enumerate(mesh.cell_sets_dict[name]):
                if len(cell_indices) > 0 and mesh.cells[block_idx].type == 'line':
                    edges = mesh.cells[block_idx].data[cell_indices]
                    for e in edges:
                        p0, p1 = points[e[0]], points[e[1]]
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=2.5)
            ax.plot([], [], color=color, linewidth=2.5, label=name)   # Legend entry

    ax.legend(loc='best')
    ax.set_aspect('equal')
    ax.set_title('Mesh with Boundary Conditions')
    plt.tight_layout()
    plt.show()

# plot_mesh_and_boundaries('domain_with_hole.msh')
```

## Practical Workflows

### 1. Classic FEM Domain: Channel with Obstacle

```python
import gmsh

def create_channel_with_obstacle(
    length: float = 8.0,
    height: float = 2.0,
    obstacle_cx: float = 2.0,
    obstacle_cy: float = 1.0,
    obstacle_r: float = 0.3,
    mesh_size_far: float = 0.2,
    mesh_size_near: float = 0.05,
    output: str = 'channel.msh'
):
    """
    Standard CFD benchmark: 2D channel with a cylindrical obstacle.
    Physical groups: inlet (left), outlet (right), walls (top+bottom), obstacle (circle).
    Adaptive mesh refinement around the obstacle.
    """
    gmsh.initialize()
    try:
        # Geometry
        channel = gmsh.model.occ.addRectangle(0, 0, 0, length, height)
        circle  = gmsh.model.occ.addDisk(obstacle_cx, obstacle_cy, 0, obstacle_r, obstacle_r)

        # Cut obstacle from channel
        out, _ = gmsh.model.occ.booleanDifference([(2, channel)], [(2, circle)])
        domain_tag = out[0][1]

        gmsh.model.occ.synchronize()

        # ─── Identify boundaries by center-of-mass ───
        bounds = gmsh.model.occ.getBoundary([(2, domain_tag)], oriented=False)

        inlet, outlet, walls, obstacle = [], [], [], []
        for dim, tag in bounds:
            cx, cy, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
            if abs(cx) < 1e-6:
                inlet.append(tag)
            elif abs(cx - length) < 1e-6:
                outlet.append(tag)
            elif abs((cx - obstacle_cx)**2 + (cy - obstacle_cy)**2 - obstacle_r**2) < obstacle_r * 0.5:
                obstacle.append(tag)
            else:
                walls.append(tag)

        # ─── Physical groups ───
        gmsh.model.mesh.addPhysicalGroup(1, inlet,    name="inlet")
        gmsh.model.mesh.addPhysicalGroup(1, outlet,   name="outlet")
        gmsh.model.mesh.addPhysicalGroup(1, walls,    name="walls")
        gmsh.model.mesh.addPhysicalGroup(1, obstacle, name="obstacle")
        gmsh.model.mesh.addPhysicalGroup(2, [domain_tag], name="domain")

        # ─── Adaptive mesh size: refine near obstacle ───
        # Distance field from obstacle curves
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", obstacle)

        # Threshold: map distance to element size
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField",  1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin",  mesh_size_near)
        gmsh.model.mesh.field.setNumber(2, "SizeMax",  mesh_size_far)
        gmsh.model.mesh.field.setNumber(2, "DistMin",  0.0)
        gmsh.model.mesh.field.setNumber(2, "DistMax",  obstacle_r * 3)

        gmsh.model.mesh.field.setAsBackground(2)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)

        # ─── Mesh ───
        gmsh.model.mesh.generate(2)
        gmsh.write(output)

        # Stats
        nodes, _ = gmsh.model.mesh.getNodes()
        print(f"Generated mesh: {len(nodes)} nodes → {output}")

    finally:
        gmsh.finalize()

# create_channel_with_obstacle()
```

### 2. 3D Geometry: Bracket with Bolt Holes

```python
import gmsh

def create_bracket(output: str = 'bracket.msh'):
    """
    3D structural bracket: rectangular plate with two through-holes.
    Physical groups: all 6 outer faces tagged, hole surfaces tagged,
    one face tagged as "fixed" for structural BC.
    """
    gmsh.initialize()
    try:
        # Base plate
        plate = gmsh.model.occ.addBox(0, 0, 0, 10, 6, 2)

        # Two cylindrical holes
        hole1 = gmsh.model.occ.addCylinder(2.0, 3.0, -1, 0, 0, 4, radius=1.0)
        hole2 = gmsh.model.occ.addCylinder(8.0, 3.0, -1, 0, 0, 4, radius=1.0)

        # Cut holes
        out, _ = gmsh.model.occ.booleanDifference(
            [(3, plate)],
            [(3, hole1), (3, hole2)]
        )
        solid_tag = out[0][1]

        gmsh.model.occ.synchronize()

        # ─── Tag surfaces by position ───
        surfaces = gmsh.model.getBoundary([(3, solid_tag)], dim=True, oriented=False)

        fixed_surfs, load_surfs, hole_surfs, other_surfs = [], [], [], []

        for dim, tag in surfaces:
            bb = gmsh.model.getBoundingBox(dim, tag)
            xmin, ymin, zmin, xmax, ymax, zmax = bb
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            cz = (zmin + zmax) / 2

            # Identify by bounding box analysis
            if abs(xmin) < 1e-6 and abs(xmax) < 1e-6:       # x=0 face → fixed
                fixed_surfs.append(tag)
            elif abs(xmin - 10) < 1e-6 and abs(xmax - 10) < 1e-6:  # x=10 face → load
                load_surfs.append(tag)
            elif abs(zmax - zmin - 2.0) > 0.01:              # Not a flat plate face → hole
                hole_surfs.append(tag)
            else:
                other_surfs.append(tag)

        # Physical groups
        if fixed_surfs: gmsh.model.mesh.addPhysicalGroup(2, fixed_surfs, name="fixed")
        if load_surfs:  gmsh.model.mesh.addPhysicalGroup(2, load_surfs,  name="load")
        if hole_surfs:  gmsh.model.mesh.addPhysicalGroup(2, hole_surfs,  name="holes")
        gmsh.model.mesh.addPhysicalGroup(3, [solid_tag], name="solid")

        # ─── Mesh size: refine near holes ───
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3)

        if hole_surfs:
            gmsh.model.mesh.field.add("Distance", 1)
            gmsh.model.mesh.field.setNumbers(1, "SurfacesList", hole_surfs)
            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "InField",  1)
            gmsh.model.mesh.field.setNumber(2, "SizeMin",  0.2)
            gmsh.model.mesh.field.setNumber(2, "SizeMax",  1.0)
            gmsh.model.mesh.field.setNumber(2, "DistMin",  0.0)
            gmsh.model.mesh.field.setNumber(2, "DistMax",  2.0)
            gmsh.model.mesh.field.setAsBackground(2)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)

        # ─── Generate 3D tet mesh ───
        gmsh.model.mesh.generate(3)
        gmsh.write(output)

        nodes, _ = gmsh.model.mesh.getNodes()
        print(f"3D mesh: {len(nodes)} nodes → {output}")

    finally:
        gmsh.finalize()

# create_bracket()
```

### 3. Mesh Conversion and Validation Pipeline

```python
import meshio
import numpy as np

def validate_and_convert(input_path: str,
                         output_formats: list[str] = ['vtu', 'xdmf'],
                         expected_groups: list[str] = None) -> dict:
    """
    Read a mesh, validate structure, convert to multiple formats.
    Returns a report of the mesh contents.
    """
    mesh = meshio.read(input_path)

    report = {
        'source_file': input_path,
        'num_nodes':   mesh.points.shape[0],
        'cell_types':  {},
        'physical_groups': [],
        'bounding_box':    {},
        'warnings':        [],
        'outputs':         []
    }

    # ─── Cell type counts ───
    for ctype, cells in mesh.cells_dict.items():
        report['cell_types'][ctype] = cells.shape[0]

    # ─── Physical groups ───
    if mesh.cell_sets_dict:
        for name in mesh.cell_sets_dict:
            total = sum(len(idx) for idx in mesh.cell_sets_dict[name])
            report['physical_groups'].append({'name': name, 'elements': total})

    # ─── Expected groups check ───
    if expected_groups:
        found = {g['name'] for g in report['physical_groups']}
        missing = set(expected_groups) - found
        if missing:
            report['warnings'].append(f"Missing physical groups: {missing}")

    # ─── Bounding box ───
    mins = mesh.points.min(axis=0)
    maxs = mesh.points.max(axis=0)
    report['bounding_box'] = {
        'x': (float(mins[0]), float(maxs[0])),
        'y': (float(mins[1]), float(maxs[1])),
        'z': (float(mins[2]), float(maxs[2]))
    }

    # ─── Quality check: degenerate elements ───
    if 'triangle' in mesh.cells_dict:
        tris = mesh.cells_dict['triangle']
        pts  = mesh.points
        areas = []
        for tri in tris:
            v0, v1, v2 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            areas.append(area)
        areas = np.array(areas)
        degenerate = np.sum(areas < 1e-12)
        if degenerate > 0:
            report['warnings'].append(f"{degenerate} degenerate triangles (area ≈ 0)")
        report['cell_types']['triangle_min_area'] = float(areas.min())
        report['cell_types']['triangle_mean_area'] = float(areas.mean())

    # ─── Convert ───
    base = input_path.rsplit('.', 1)[0]
    for fmt in output_formats:
        out_path = f"{base}.{fmt}"
        try:
            mesh.write(out_path)
            report['outputs'].append(out_path)
        except Exception as e:
            report['warnings'].append(f"Failed to write {fmt}: {e}")

    # ─── Print report ───
    print(f"\n{'='*50}")
    print(f"  MESH REPORT: {input_path}")
    print(f"{'='*50}")
    print(f"  Nodes:           {report['num_nodes']:>10,}")
    for ctype, count in report['cell_types'].items():
        if isinstance(count, int):
            print(f"  {ctype:16s}: {count:>10,}")
    print(f"\n  Physical groups:")
    for g in report['physical_groups']:
        print(f"    {g['name']:20s}: {g['elements']:>8,} elements")
    if report['warnings']:
        print(f"\n  ⚠️  Warnings:")
        for w in report['warnings']:
            print(f"    {w}")
    print(f"\n  Outputs: {report['outputs']}")

    return report

# report = validate_and_convert(
#     'channel.msh',
#     output_formats=['vtu', 'xdmf'],
#     expected_groups=['inlet', 'outlet', 'walls', 'obstacle', 'domain']
# )
```

### 4. FEniCS-Ready Mesh Pipeline

```python
import gmsh
import meshio
import numpy as np

def fenics_ready_mesh(geometry_func,
                      output_msh: str = 'fenics_mesh.msh',
                      output_xdmf: str = 'fenics_mesh.xdmf'):
    """
    Full pipeline: geometry → mesh → validate → convert to XDMF (FEniCS native format).

    geometry_func: callable that builds geometry inside an initialized gmsh session.
                   Must define physical groups.
    """
    # 1. Generate mesh
    gmsh.initialize()
    try:
        geometry_func()   # User-provided: builds geometry + physical groups
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(output_msh)

        # Print mesh info from gmsh directly
        nodes, _ = gmsh.model.mesh.getNodes()
        print(f"Mesh generated: {len(nodes)} nodes")
    finally:
        gmsh.finalize()

    # 2. Read with meshio, strip gmsh-specific cruft, write XDMF
    mesh = meshio.read(output_msh)

    # Remove higher-order elements if present (keep only linear)
    keep_types = {'triangle', 'tetra', 'line', 'vertex',
                  'triangle6', 'tetra10'}   # Allow quadratic too
    mesh.cells = [block for block in mesh.cells if block.type in keep_types]

    # Remove point/cell data that FEniCS doesn't need
    mesh.point_data = {}
    # Keep cell_sets (physical groups) — these are the boundary conditions

    # Write XDMF (FEniCS native)
    mesh.write(output_xdmf, file_format='xdmf')
    print(f"XDMF written: {output_xdmf}")
    print(f"Physical groups: {list(mesh.cell_sets_dict.keys())}")

    return mesh

# ─── Example usage ───
def my_geometry():
    """Define a unit square with inlet/outlet/walls."""
    rect = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()

    bounds = gmsh.model.occ.getBoundary([(2, rect)], oriented=False)
    left, right, top_bottom = [], [], []
    for dim, tag in bounds:
        cx, cy, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(cx) < 1e-6:       left.append(tag)
        elif abs(cx - 1) < 1e-6: right.append(tag)
        else:                    top_bottom.append(tag)

    gmsh.model.mesh.addPhysicalGroup(1, left,       name="inlet")
    gmsh.model.mesh.addPhysicalGroup(1, right,      name="outlet")
    gmsh.model.mesh.addPhysicalGroup(1, top_bottom, name="walls")
    gmsh.model.mesh.addPhysicalGroup(2, [rect],     name="domain")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)

# mesh = fenics_ready_mesh(my_geometry)
```

## Common Pitfalls and Solutions

### Empty Mesh After generate()

```python
import gmsh

# ❌ PROBLEM: synchronize() missing → mesher sees nothing
gmsh.initialize()
gmsh.model.occ.addBox(0,0,0,1,1,1)
gmsh.model.mesh.generate(3)
# → 0 nodes, 0 elements

# ✅ SOLUTION: synchronize() AFTER all geometry, BEFORE mesh
gmsh.initialize()
gmsh.model.occ.addBox(0,0,0,1,1,1)
gmsh.model.occ.synchronize()     # ← This line
gmsh.model.mesh.generate(3)
# → Correct mesh
gmsh.finalize()
```

### Physical Groups Missing from meshio Output

```python
import gmsh
import meshio

# ❌ PROBLEM: Physical groups defined on wrong dimension
gmsh.initialize()
gmsh.model.occ.addRectangle(0,0,0,1,1)
gmsh.model.occ.synchronize()
# "inlet" is a CURVE (dim=1), but user tags it as surface (dim=2):
gmsh.model.mesh.addPhysicalGroup(2, [1], name="inlet")   # dim=2 = surface ← WRONG for a curve!
gmsh.model.mesh.generate(2)
gmsh.write('bad.msh')
gmsh.finalize()

mesh = meshio.read('bad.msh')
# "inlet" won't contain the boundary curves you expect

# ✅ SOLUTION: Match dimension to entity type
# Curves → dim=1, Surfaces → dim=2, Volumes → dim=3
gmsh.initialize()
rect = gmsh.model.occ.addRectangle(0,0,0,1,1)
gmsh.model.occ.synchronize()
bounds = gmsh.model.occ.getBoundary([(2, rect)], oriented=False)
# bounds are (dim=1, tag) — curves
inlet_tags = [tag for dim, tag in bounds if abs(gmsh.model.occ.getCenterOfMass(dim, tag)[0]) < 1e-6]
gmsh.model.mesh.addPhysicalGroup(1, inlet_tags, name="inlet")   # dim=1 ← Correct
gmsh.model.mesh.addPhysicalGroup(2, [rect], name="domain")      # dim=2 ← Correct
gmsh.model.mesh.generate(2)
gmsh.write('good.msh')
gmsh.finalize()
```

### Boolean Op Returns Unexpected Tags

```python
import gmsh

# ❌ PROBLEM: Using pre-boolean tags after the operation
gmsh.initialize()
box  = gmsh.model.occ.addBox(0,0,0,2,2,2)   # tag = 1
hole = gmsh.model.occ.addCylinder(1,1,-1,0,0,4,0.5)  # tag = 1 (different dim!)

gmsh.model.occ.booleanDifference([(3, box)], [(3, hole)])
gmsh.model.occ.synchronize()

# box (tag=1) no longer exists as-is. Using tag=1 for the result is LUCK, not correctness.
gmsh.model.mesh.addPhysicalGroup(3, [1], name="solid")   # May or may not work!

# ✅ SOLUTION: Always capture and use the return value
gmsh.initialize()
box  = gmsh.model.occ.addBox(0,0,0,2,2,2)
hole = gmsh.model.occ.addCylinder(1,1,-1,0,0,4,0.5)

out, _ = gmsh.model.occ.booleanDifference([(3, box)], [(3, hole)])
result_tag = out[0][1]   # THE correct tag of the result

gmsh.model.occ.synchronize()
gmsh.model.mesh.addPhysicalGroup(3, [result_tag], name="solid")   # ← Guaranteed correct
gmsh.model.mesh.generate(3)
gmsh.write('correct.msh')
gmsh.finalize()
```

### meshio Loses Physical Groups on Conversion

```python
import meshio

# ❌ PROBLEM: Some formats don't support physical groups natively.
# Converting .msh → .vtk may drop cell_sets entirely.
mesh = meshio.read('with_groups.msh')
mesh.write('output.vtk')   # VTK legacy: cell_sets may be lost

mesh2 = meshio.read('output.vtk')
print(mesh2.cell_sets_dict)   # May be empty!

# ✅ SOLUTION 1: Use formats that preserve cell sets
mesh.write('output.xdmf')    # XDMF preserves cell sets ← FEniCS standard
mesh.write('output.vtu')     # VTU (XML) preserves cell sets better than legacy VTK

# ✅ SOLUTION 2: Encode physical groups as cell_data before writing VTK
import numpy as np
# Add a 'physical_group_id' field to each element
# This survives any format conversion
physical_ids = np.zeros(sum(len(c.data) for c in mesh.cells), dtype=int)
# ... fill in IDs based on cell_sets ...
mesh.cell_data['physical_group'] = [physical_ids]
mesh.write('output.vtk')   # physical_group field survives
```

---

gmsh + meshio is the geometry layer every simulation needs. gmsh builds the shape and controls the mesh density; meshio moves the result into whatever format the solver demands. The critical discipline is the **synchronize → physical groups → mesh → convert** sequence: each step depends on the previous one being correct, and skipping any one silently breaks everything downstream. Get this pipeline right and you have a reusable geometry factory for any FEM workflow.