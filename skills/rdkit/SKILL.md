---
name: rdkit
description: Open-source cheminformatics and machine learning toolkit for drug discovery, molecular manipulation, and chemical property calculation. RDKit handles SMILES, molecular fingerprints, substructure searching, 3D conformer generation, pharmacophore modeling, and QSAR. Use when working with chemical structures, drug-like properties, molecular similarity, virtual screening, or computational chemistry workflows.
version: 2023.09.4
license: BSD-3-Clause
---

# RDKit - Cheminformatics and Drug Discovery

RDKit is the industry-standard open-source toolkit for cheminformatics. It provides comprehensive tools for molecular manipulation, descriptor calculation, fingerprinting, substructure searching, and 3D molecular modeling. RDKit is used extensively in pharmaceutical companies for drug discovery and virtual screening.

## When to Use

- Reading and writing chemical file formats (SMILES, SDF, MOL2, PDB).
- Calculating molecular descriptors and drug-like properties (Lipinski's Rule of Five).
- Generating molecular fingerprints for similarity searching.
- Substructure searching and chemical pattern matching (SMARTS).
- 3D conformer generation and molecular alignment.
- Virtual screening of compound libraries.
- Pharmacophore modeling and shape similarity.
- QSAR (Quantitative Structure-Activity Relationship) modeling.
- Reaction enumeration and retrosynthesis.
- Visualizing chemical structures in 2D and 3D.
- Building machine learning models for molecular property prediction.

## Reference Documentation

**Official docs**: https://www.rdkit.org/docs/  
**RDKit Book**: https://www.rdkit.org/docs/RDKit_Book.html  
**GitHub**: https://github.com/rdkit/rdkit  
**Search patterns**: `rdkit.Chem`, `rdkit.Chem.Descriptors`, `rdkit.Chem.AllChem`, `rdkit.DataStructs`

## Core Principles

### Molecular Representation
RDKit represents molecules as graphs where atoms are nodes and bonds are edges. The core object is `Mol`, which can be created from SMILES, SDF files, or built programmatically.

### SMILES (Simplified Molecular Input Line Entry System)
A text-based notation for chemical structures. Example: `CCO` is ethanol, `c1ccccc1` is benzene. RDKit can parse and generate SMILES strings.

### Fingerprints for Similarity
Molecular fingerprints are binary vectors encoding structural features. They enable fast similarity searching and clustering of large compound libraries.

### Lazy Evaluation
Many RDKit operations are lazy - properties are computed only when needed. This makes operations on large libraries very efficient.

## Quick Reference

### Installation

```bash
# Via conda (recommended)
conda install -c conda-forge rdkit

# Via pip
pip install rdkit

# For visualization
pip install rdkit pillow
```

### Standard Imports

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np
import pandas as pd
```

### Basic Pattern - SMILES to Molecule

```python
from rdkit import Chem

# 1. Create molecule from SMILES
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
mol = Chem.MolFromSmiles(smiles)

# 2. Check if molecule is valid
if mol is None:
    print("Invalid SMILES")
else:
    print(f"Molecular formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
    print(f"Molecular weight: {Descriptors.MolWt(mol):.2f}")
```

### Basic Pattern - Calculate Properties

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")

# Calculate drug-like properties
mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
hbd = Lipinski.NumHDonors(mol)
hba = Lipinski.NumHAcceptors(mol)

print(f"MW: {mw:.2f}, LogP: {logp:.2f}, HBD: {hbd}, HBA: {hba}")

# Check Lipinski's Rule of Five
lipinski_pass = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
print(f"Lipinski compliant: {lipinski_pass}")
```

## Critical Rules

### ✅ DO

- **Always Validate Molecules** - Check `mol is not None` after parsing SMILES/files to catch invalid structures.
- **Use Canonical SMILES** - Use `Chem.MolToSmiles(mol)` to get canonical (standardized) SMILES for comparison.
- **Sanitize Molecules** - RDKit auto-sanitizes by default (valence checking, aromaticity). Keep it enabled unless you have a specific reason.
- **Generate 3D Coordinates** - Use `AllChem.EmbedMolecule()` before 3D operations like alignment or docking.
- **Use Fingerprints for Large Libraries** - For similarity searching in millions of compounds, fingerprints are 1000x faster than direct comparison.
- **Specify Random Seeds** - For reproducible conformer generation, always set `randomSeed`.
- **Handle Stereochemistry** - Use `Chem.AssignStereochemistry()` to properly assign R/S and E/Z labels.
- **Batch Processing** - Use generators or chunking for processing millions of molecules to avoid memory issues.

### ❌ DON'T

- **Don't Ignore Invalid Molecules** - Always handle the case when `MolFromSmiles()` returns `None`.
- **Don't Compare SMILES Strings Directly** - Two different SMILES can represent the same molecule. Use canonical SMILES or InChI.
- **Don't Skip Kekulization** - For aromatic systems, ensure proper Kekulé structure assignment.
- **Don't Use Descriptors for Similarity** - Use fingerprints (much faster and more appropriate).
- **Don't Forget Hydrogens** - Add explicit hydrogens with `Chem.AddHs()` when needed for 3D operations.
- **Don't Overuse 3D Minimization** - Energy minimization is slow; only use when necessary (docking, visualization).

## Anti-Patterns (NEVER)

```python
from rdkit import Chem
from rdkit.Chem import AllChem

# ❌ BAD: Not checking if molecule is valid
smiles = "INVALID_SMILES"
mol = Chem.MolFromSmiles(smiles)
mw = Descriptors.MolWt(mol)  # Crashes!

# ✅ GOOD: Always validate
mol = Chem.MolFromSmiles(smiles)
if mol is not None:
    mw = Descriptors.MolWt(mol)
else:
    print("Invalid SMILES")

# ❌ BAD: Comparing SMILES strings directly
smiles1 = "CC(C)C"  # isobutane
smiles2 = "C(C)CC"  # same molecule, different SMILES
if smiles1 == smiles2:  # False, but same molecule!
    print("Same")

# ✅ GOOD: Use canonical SMILES
mol1 = Chem.MolFromSmiles(smiles1)
mol2 = Chem.MolFromSmiles(smiles2)
can1 = Chem.MolToSmiles(mol1)
can2 = Chem.MolToSmiles(mol2)
if can1 == can2:  # True
    print("Same molecule")

# ❌ BAD: 3D operations without 3D coordinates
mol = Chem.MolFromSmiles("CCO")
AllChem.AlignMol(mol, ref_mol)  # Fails! No 3D coords

# ✅ GOOD: Generate 3D coordinates first
mol = Chem.MolFromSmiles("CCO")
AllChem.EmbedMolecule(mol)
AllChem.AlignMol(mol, ref_mol)
```

## Molecular I/O and Conversion

### SMILES Parsing

```python
from rdkit import Chem

# Parse SMILES
mol = Chem.MolFromSmiles("CCO")

# Parse SMILES with sanitization control
mol = Chem.MolFromSmiles("CCO", sanitize=True)  # Default

# Generate canonical SMILES
canonical = Chem.MolToSmiles(mol)

# Generate isomeric SMILES (includes stereochemistry)
iso_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

# Generate SMILES without stereochemistry
non_iso = Chem.MolToSmiles(mol, isomericSmiles=False)

# Handle invalid SMILES
smiles_list = ["CCO", "INVALID", "c1ccccc1"]
mols = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        mols.append(mol)
    else:
        print(f"Failed to parse: {smi}")
```

### Reading SDF Files

```python
from rdkit import Chem

# Read single molecule from file
mol = Chem.MolFromMolFile("molecule.mol")

# Read multiple molecules from SDF
suppl = Chem.SDMolSupplier("compounds.sdf")

# Iterate through molecules
for mol in suppl:
    if mol is None:
        continue
    
    smiles = Chem.MolToSmiles(mol)
    print(f"SMILES: {smiles}")
    
    # Access SDF properties
    if mol.HasProp("_Name"):
        name = mol.GetProp("_Name")
        print(f"Name: {name}")

# Read with removeHs=False to keep explicit hydrogens
suppl = Chem.SDMolSupplier("compounds.sdf", removeHs=False)
```

### Writing SDF Files

```python
from rdkit import Chem

# Write single molecule
mol = Chem.MolFromSmiles("CCO")
writer = Chem.SDWriter("output.sdf")
writer.write(mol)
writer.close()

# Write multiple molecules
mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CC(=O)O"]]
writer = Chem.SDWriter("output.sdf")
for mol in mols:
    if mol is not None:
        writer.write(mol)
writer.close()

# Add properties to molecules
mol = Chem.MolFromSmiles("CCO")
mol.SetProp("_Name", "Ethanol")
mol.SetProp("Activity", "10.5")
writer = Chem.SDWriter("output.sdf")
writer.write(mol)
writer.close()
```

### InChI and InChIKey

```python
from rdkit import Chem

mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin

# Generate InChI (unique chemical identifier)
inchi = Chem.MolToInchi(mol)
print(f"InChI: {inchi}")

# Generate InChIKey (hashed InChI, good for database lookups)
inchikey = Chem.MolToInchiKey(mol)
print(f"InChIKey: {inchikey}")

# Parse InChI
mol_from_inchi = Chem.MolFromInchi(inchi)
```

## Molecular Descriptors

### Common Descriptors

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen

mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin

# Basic properties
mw = Descriptors.MolWt(mol)
num_atoms = mol.GetNumAtoms()
num_heavy_atoms = Lipinski.HeavyAtomCount(mol)

# Lipinski's Rule of Five parameters
logp = Descriptors.MolLogP(mol)
hbd = Lipinski.NumHDonors(mol)
hba = Lipinski.NumHAcceptors(mol)
rotatable_bonds = Lipinski.NumRotatableBonds(mol)

# Topological descriptors
tpsa = Descriptors.TPSA(mol)  # Topological polar surface area
rings = Lipinski.RingCount(mol)
aromatic_rings = Lipinski.NumAromaticRings(mol)

# Complexity
bertz_ct = Descriptors.BertzCT(mol)  # Molecular complexity

print(f"""
Molecular Weight: {mw:.2f}
LogP: {logp:.2f}
HBD: {hbd}
HBA: {hba}
TPSA: {tpsa:.2f}
Rotatable Bonds: {rotatable_bonds}
Aromatic Rings: {aromatic_rings}
""")
```

### Calculate All Descriptors

```python
from rdkit import Chem
from rdkit.Chem import Descriptors

mol = Chem.MolFromSmiles("CCO")

# Get all available descriptors
descriptor_names = [desc[0] for desc in Descriptors.descList]

# Calculate all descriptors
descriptors = {}
for name in descriptor_names:
    calc = getattr(Descriptors, name)
    descriptors[name] = calc(mol)

print(f"Total descriptors: {len(descriptors)}")
print(f"First 5: {list(descriptors.items())[:5]}")
```

### Drug-Likeness Filters

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def check_lipinski(mol):
    """Check Lipinski's Rule of Five."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    
    rules = {
        'MW <= 500': mw <= 500,
        'LogP <= 5': logp <= 5,
        'HBD <= 5': hbd <= 5,
        'HBA <= 10': hba <= 10
    }
    
    passed = all(rules.values())
    return passed, rules

def check_veber(mol):
    """Check Veber's rules for oral bioavailability."""
    rotatable = Lipinski.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    
    rules = {
        'Rotatable bonds <= 10': rotatable <= 10,
        'TPSA <= 140': tpsa <= 140
    }
    
    passed = all(rules.values())
    return passed, rules

# Usage
mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
lipinski_pass, lipinski_rules = check_lipinski(mol)
veber_pass, veber_rules = check_veber(mol)

print(f"Lipinski: {lipinski_pass}")
print(f"Veber: {veber_pass}")
```

## Molecular Fingerprints

### Morgan Fingerprints (Circular)

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# Create molecule
mol = Chem.MolFromSmiles("CCO")

# Generate Morgan fingerprint (ECFP4)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# Convert to numpy array
import numpy as np
arr = np.zeros((1,))
DataStructs.ConvertToNumpyArray(fp, arr)

# Generate count-based fingerprint (for feature importance)
fp_counts = AllChem.GetMorganFingerprint(mol, radius=2)

# Get feature info
info = {}
fp_info = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=info)
print(f"Number of on-bits: {len(info)}")
```

### Fingerprint Similarity

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# Create molecules
mol1 = Chem.MolFromSmiles("CCO")
mol2 = Chem.MolFromSmiles("CCCO")
mol3 = Chem.MolFromSmiles("c1ccccc1")

# Generate fingerprints
fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2)
fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2)
fp3 = AllChem.GetMorganFingerprintAsBitVect(mol3, radius=2)

# Calculate Tanimoto similarity
sim_12 = DataStructs.TanimotoSimilarity(fp1, fp2)
sim_13 = DataStructs.TanimotoSimilarity(fp1, fp3)

print(f"Ethanol vs Propanol: {sim_12:.3f}")  # High similarity
print(f"Ethanol vs Benzene: {sim_13:.3f}")   # Low similarity

# Calculate Dice similarity
dice_12 = DataStructs.DiceSimilarity(fp1, fp2)

# Bulk similarity (compare one to many)
fps = [fp1, fp2, fp3]
similarities = DataStructs.BulkTanimotoSimilarity(fp1, fps)
print(f"Bulk similarities: {similarities}")
```

### Other Fingerprint Types

```python
from rdkit import Chem
from rdkit.Chem import AllChem, RDKFingerprint

mol = Chem.MolFromSmiles("CCO")

# RDKit fingerprint (topological)
fp_rdkit = Chem.RDKFingerprint(mol)

# Atom pair fingerprint
fp_atompair = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol)

# Topological torsion fingerprint
fp_torsion = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)

# MACCS keys (166-bit structural keys)
from rdkit.Chem import MACCSkeys
fp_maccs = MACCSkeys.GenMACCSKeys(mol)
```

## Substructure Searching

### SMARTS Pattern Matching

```python
from rdkit import Chem

# Define molecule
mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin

# SMARTS pattern for carboxylic acid
pattern = Chem.MolFromSmarts("C(=O)O")

# Check if substructure exists
has_match = mol.HasSubstructMatch(pattern)
print(f"Contains carboxylic acid: {has_match}")

# Get matching atoms
matches = mol.GetSubstructMatches(pattern)
print(f"Number of matches: {len(matches)}")
print(f"Matching atom indices: {matches}")

# Common SMARTS patterns
patterns = {
    'Carboxylic acid': 'C(=O)O',
    'Ester': 'C(=O)O[C,c]',
    'Amide': 'C(=O)N',
    'Alcohol': '[OH][C,c]',
    'Primary amine': '[NH2][C,c]',
    'Aromatic ring': 'c1ccccc1'
}

for name, smarts in patterns.items():
    pattern = Chem.MolFromSmarts(smarts)
    if mol.HasSubstructMatch(pattern):
        print(f"Contains {name}")
```

### Substructure Filtering

```python
from rdkit import Chem

# Library of molecules
smiles_list = [
    "CC(=O)O",           # Acetic acid
    "CCO",               # Ethanol
    "c1ccccc1C(=O)O",    # Benzoic acid
    "CCCC",              # Butane
]

mols = [Chem.MolFromSmiles(s) for s in smiles_list]

# Filter molecules containing carboxylic acid
pattern = Chem.MolFromSmarts("C(=O)O")
filtered = [mol for mol in mols if mol.HasSubstructMatch(pattern)]

print(f"Molecules with carboxylic acid: {len(filtered)}/{len(mols)}")

# Filter by multiple patterns (AND logic)
pattern1 = Chem.MolFromSmarts("c1ccccc1")  # Aromatic ring
pattern2 = Chem.MolFromSmarts("C(=O)O")    # Carboxylic acid

aromatic_acids = [
    mol for mol in mols
    if mol.HasSubstructMatch(pattern1) and mol.HasSubstructMatch(pattern2)
]
```

### Replace Substructures

```python
from rdkit import Chem
from rdkit.Chem import AllChem

# Replace carboxylic acid with ester
mol = Chem.MolFromSmiles("CC(=O)O")  # Acetic acid

# Define replacement
rxn = AllChem.ReactionFromSmarts('[C:1](=O)O>>[C:1](=O)OC')

# Apply reaction
products = rxn.RunReactants((mol,))

if products:
    product = products[0][0]
    print(f"Product: {Chem.MolToSmiles(product)}")  # Methyl acetate
```

## 3D Conformer Generation

### Generate 3D Coordinates

```python
from rdkit import Chem
from rdkit.Chem import AllChem

# Create molecule
mol = Chem.MolFromSmiles("CCO")

# Add hydrogens (required for 3D)
mol = Chem.AddHs(mol)

# Generate 3D coordinates
result = AllChem.EmbedMolecule(mol, randomSeed=42)

if result == 0:  # Success
    print("3D coordinates generated")
else:
    print("Failed to generate 3D coordinates")

# Optimize geometry with MMFF force field
AllChem.MMFFOptimizeMolecule(mol)

# Get atomic positions
conf = mol.GetConformer()
for i in range(mol.GetNumAtoms()):
    pos = conf.GetAtomPosition(i)
    print(f"Atom {i}: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
```

### Multiple Conformers

```python
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles("CCCC")  # Butane
mol = Chem.AddHs(mol)

# Generate multiple conformers
conf_ids = AllChem.EmbedMultipleConfs(
    mol,
    numConfs=10,
    randomSeed=42,
    pruneRmsThresh=0.5  # Remove similar conformers
)

print(f"Generated {len(conf_ids)} conformers")

# Optimize each conformer
for conf_id in conf_ids:
    AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)

# Get energies
props = AllChem.MMFFGetMoleculeProperties(mol)
for conf_id in conf_ids:
    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    energy = ff.CalcEnergy()
    print(f"Conformer {conf_id}: {energy:.2f} kcal/mol")
```

### Molecular Alignment

```python
from rdkit import Chem
from rdkit.Chem import AllChem

# Reference molecule
ref_mol = Chem.MolFromSmiles("c1ccccc1C")  # Toluene
ref_mol = Chem.AddHs(ref_mol)
AllChem.EmbedMolecule(ref_mol)

# Probe molecule
probe_mol = Chem.MolFromSmiles("c1ccccc1CC")  # Ethylbenzene
probe_mol = Chem.AddHs(probe_mol)
AllChem.EmbedMolecule(probe_mol)

# Align probe to reference
rmsd = AllChem.AlignMol(probe_mol, ref_mol)
print(f"RMSD: {rmsd:.3f} Å")

# Get aligned coordinates
# Now probe_mol has coordinates aligned to ref_mol
```

## Molecular Visualization

### 2D Drawings

```python
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

# Single molecule
mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
img = Draw.MolToImage(mol, size=(300, 300))
plt.imshow(img)
plt.axis('off')
plt.show()

# Multiple molecules
mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CC(=O)O"]]
legends = ["Ethanol", "Benzene", "Acetic acid"]

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=3,
    subImgSize=(200, 200),
    legends=legends
)
plt.imshow(img)
plt.axis('off')
plt.show()
```

### Highlight Substructures

```python
from rdkit import Chem
from rdkit.Chem import Draw

mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin

# Highlight carboxylic acid group
pattern = Chem.MolFromSmarts("C(=O)O")
match = mol.GetSubstructMatch(pattern)

# Draw with highlighted atoms
img = Draw.MolToImage(mol, highlightAtoms=match, size=(300, 300))
```

### Save to File

```python
from rdkit import Chem
from rdkit.Chem import Draw

mol = Chem.MolFromSmiles("CCO")

# Save as PNG
Draw.MolToFile(mol, "molecule.png", size=(300, 300))

# Save as SVG (vector graphics)
from rdkit.Chem.Draw import rdMolDraw2D

drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
drawer.DrawMolecule(mol)
drawer.FinishDrawing()
svg = drawer.GetDrawingText()

with open("molecule.svg", "w") as f:
    f.write(svg)
```

## Practical Workflows

### 1. Virtual Screening Pipeline

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit import DataStructs
import pandas as pd

def screen_library(library_file, reference_smiles, similarity_threshold=0.7):
    """Screen compound library for similar, drug-like molecules."""
    
    # Reference molecule and fingerprint
    ref_mol = Chem.MolFromSmiles(reference_smiles)
    ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, radius=2)
    
    # Results
    hits = []
    
    # Read library
    suppl = Chem.SDMolSupplier(library_file)
    
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        
        # Step 1: Drug-likeness filter (Lipinski)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        if not (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10):
            continue
        
        # Step 2: Similarity filter
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2)
        similarity = DataStructs.TanimotoSimilarity(ref_fp, fp)
        
        if similarity < similarity_threshold:
            continue
        
        # Step 3: PAINS filter (Pan-Assay Interference Compounds)
        from rdkit.Chem import FilterCatalog
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)
        
        if catalog.HasMatch(mol):
            continue
        
        # Passed all filters
        hits.append({
            'id': i,
            'smiles': Chem.MolToSmiles(mol),
            'similarity': similarity,
            'mw': mw,
            'logp': logp
        })
    
    # Convert to DataFrame
    df_hits = pd.DataFrame(hits)
    df_hits = df_hits.sort_values('similarity', ascending=False)
    
    return df_hits

# Usage
# hits = screen_library('compounds.sdf', 'CC(=O)OC1=CC=CC=C1C(=O)O')
# print(f"Found {len(hits)} hits")
```

### 2. Diversity Selection

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

def select_diverse_set(smiles_list, n_select=100):
    """Select diverse subset using MaxMin algorithm."""
    
    # Generate fingerprints
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2) for m in mols if m]
    
    if len(fps) < n_select:
        return list(range(len(fps)))
    
    # MaxMin diversity picking
    from rdkit.SimDivFilters import MaxMinPicker
    
    def distance_function(i, j):
        return 1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
    
    picker = MaxMinPicker()
    picks = picker.LazyPick(
        distance_function,
        len(fps),
        n_select,
        seed=42
    )
    
    return list(picks)

# Usage
smiles_list = ["CCO", "CCCO", "c1ccccc1", "CC(=O)O", "CCCCCCCC"]
diverse_indices = select_diverse_set(smiles_list, n_select=3)
diverse_smiles = [smiles_list[i] for i in diverse_indices]
print(f"Selected: {diverse_smiles}")
```

### 3. QSAR Model Building

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def build_qsar_model(smiles_list, activities):
    """Build QSAR model from SMILES and activities."""
    
    # Generate fingerprints
    fps = []
    valid_activities = []
    
    for smi, act in zip(smiles_list, activities):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # Morgan fingerprint as features
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_activities.append(act)
    
    X = np.array(fps)
    y = np.array(valid_activities)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print(f"Train R²: {r2_score(y_train, y_pred_train):.3f}")
    print(f"Test R²: {r2_score(y_test, y_pred_test):.3f}")
    print(f"Test RMSE: {mean_squared_error(y_test, y_pred_test, squared=False):.3f}")
    
    return model

# Usage
# smiles = ["CCO", "CCCO", "c1ccccc1", "CC(=O)O"]
# activities = [5.2, 4.8, 6.1, 5.5]  # pIC50 values
# model = build_qsar_model(smiles, activities)
```

### 4. Scaffold Analysis

```python
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter

def analyze_scaffolds(smiles_list):
    """Analyze Murcko scaffolds in compound set."""
    
    scaffolds = []
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # Get Murcko scaffold
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold)
            scaffolds.append(scaffold_smi)
    
    # Count scaffolds
    scaffold_counts = Counter(scaffolds)
    
    print(f"Unique scaffolds: {len(scaffold_counts)}")
    print("\nTop 5 scaffolds:")
    for scaffold, count in scaffold_counts.most_common(5):
        print(f"{scaffold}: {count}")
    
    return scaffold_counts

# Usage
# smiles_list = ["c1ccccc1CC", "c1ccccc1CCC", "c1ccc(O)cc1"]
# scaffolds = analyze_scaffolds(smiles_list)
```

### 5. Reaction Enumeration

```python
from rdkit import Chem
from rdkit.Chem import AllChem

def enumerate_amide_coupling(acids, amines):
    """Enumerate all possible amide products."""
    
    # Define reaction SMARTS
    rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])O.[N:3]>>[C:1](=[O:2])[N:3]')
    
    products = []
    
    for acid_smi in acids:
        for amine_smi in amines:
            acid = Chem.MolFromSmiles(acid_smi)
            amine = Chem.MolFromSmiles(amine_smi)
            
            if acid is None or amine is None:
                continue
            
            # Run reaction
            products_tuple = rxn.RunReactants((acid, amine))
            
            if products_tuple:
                product = products_tuple[0][0]
                Chem.SanitizeMol(product)
                product_smi = Chem.MolToSmiles(product)
                products.append({
                    'acid': acid_smi,
                    'amine': amine_smi,
                    'product': product_smi
                })
    
    return products

# Usage
acids = ["CC(=O)O", "c1ccccc1C(=O)O"]
amines = ["CCN", "c1ccccc1N"]
products = enumerate_amide_coupling(acids, amines)
print(f"Generated {len(products)} amides")
```

## Performance Optimization

### Bulk Operations

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# Instead of loop
# ❌ SLOW
# fps = []
# for smi in smiles_list:
#     mol = Chem.MolFromSmiles(smi)
#     fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
#     fps.append(fp)

# ✅ FAST: Use PandasTools for bulk operations
from rdkit.Chem import PandasTools

df = pd.DataFrame({'SMILES': smiles_list})
PandasTools.AddMoleculeColumnToFrame(df, 'SMILES', 'Molecule')

# Calculate properties in bulk
df['MW'] = df['Molecule'].apply(lambda x: Descriptors.MolWt(x) if x else None)
df['LogP'] = df['Molecule'].apply(lambda x: Descriptors.MolLogP(x) if x else None)
```

### Parallel Processing

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool
import pandas as pd

def process_molecule(smiles):
    """Process single molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return {
        'smiles': smiles,
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'fp': AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    }

def process_library_parallel(smiles_list, n_jobs=4):
    """Process library in parallel."""
    with Pool(n_jobs) as pool:
        results = pool.map(process_molecule, smiles_list)
    
    # Filter None results
    results = [r for r in results if r is not None]
    return pd.DataFrame(results)

# Usage
# df = process_library_parallel(large_smiles_list, n_jobs=8)
```

### Caching Calculations

```python
from functools import lru_cache
from rdkit import Chem
from rdkit.Chem import Descriptors

@lru_cache(maxsize=10000)
def get_mol_properties(smiles):
    """Calculate properties with caching."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol)
    }

# Repeated calls will use cache
props1 = get_mol_properties("CCO")  # Calculated
props2 = get_mol_properties("CCO")  # Cached (instant)
```

## Common Pitfalls and Solutions

### The "Invalid SMILES" Problem

Not all SMILES strings are valid.

```python
# ❌ Problem: Assuming all SMILES are valid
smiles_list = ["CCO", "INVALID", "c1ccccc1"]
mols = [Chem.MolFromSmiles(s) for s in smiles_list]
# Contains None!

# ✅ Solution: Filter invalid molecules
mols = [Chem.MolFromSmiles(s) for s in smiles_list]
valid_mols = [m for m in mols if m is not None]

# ✅ Better: Track which failed
results = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        results.append({'smiles': smi, 'mol': mol, 'valid': True})
    else:
        results.append({'smiles': smi, 'mol': None, 'valid': False})
```

### The "Stereochemistry Loss" Problem

SMILES generation can lose stereochemistry if not careful.

```python
from rdkit import Chem

# Molecule with stereochemistry
chiral_smiles = "C[C@H](O)CC"  # (S)-2-butanol
mol = Chem.MolFromSmiles(chiral_smiles)

# ❌ BAD: Lose stereochemistry
non_iso = Chem.MolToSmiles(mol, isomericSmiles=False)
print(non_iso)  # "CC(O)CC" - lost chirality!

# ✅ GOOD: Preserve stereochemistry
iso = Chem.MolToSmiles(mol, isomericSmiles=True)
print(iso)  # "C[C@H](O)CC" - preserved!
```

### The "3D Without Hydrogens" Problem

3D operations require explicit hydrogens.

```python
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles("CCO")

# ❌ BAD: Generate 3D without hydrogens
result = AllChem.EmbedMolecule(mol)
# Poor quality or failure

# ✅ GOOD: Add hydrogens first
mol_h = Chem.AddHs(mol)
result = AllChem.EmbedMolecule(mol_h)
AllChem.MMFFOptimizeMolecule(mol_h)
```

### The "Fingerprint Type Mismatch" Problem

Comparing different fingerprint types gives meaningless results.

```python
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import DataStructs

mol1 = Chem.MolFromSmiles("CCO")
mol2 = Chem.MolFromSmiles("CCCO")

# ❌ BAD: Comparing different fingerprint types
fp1_morgan = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
fp2_maccs = MACCSkeys.GenMACCSKeys(mol2)

# This will error or give nonsense!
# similarity = DataStructs.TanimotoSimilarity(fp1_morgan, fp2_maccs)

# ✅ GOOD: Use same fingerprint type
fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
```

### The "Memory Explosion" Problem

Processing millions of molecules can exhaust memory.

```python
# ❌ BAD: Load entire library into memory
suppl = Chem.SDMolSupplier('huge_library.sdf')
mols = [mol for mol in suppl]  # Out of memory!

# ✅ GOOD: Process in batches
def process_in_batches(sdf_file, batch_size=10000):
    suppl = Chem.SDMolSupplier(sdf_file)
    batch = []
    
    for mol in suppl:
        if mol is not None:
            batch.append(mol)
        
        if len(batch) >= batch_size:
            # Process batch
            yield batch
            batch = []
    
    # Process remaining
    if batch:
        yield batch

# Usage
for batch in process_in_batches('huge_library.sdf'):
    # Process each batch
    pass
```

RDKit is the cornerstone of computational drug discovery and cheminformatics. Its comprehensive toolkit for molecular manipulation, descriptor calculation, and similarity searching makes it indispensable for pharmaceutical research, virtual screening, and chemical data analysis. Master RDKit, and you'll have the power to computationally explore vast chemical spaces and accelerate drug discovery.
