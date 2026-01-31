---
name: biopython
description: Comprehensive guide for Biopython - the premier Python library for computational biology and bioinformatics. Use for DNA/RNA/protein sequence analysis, file I/O (FASTA, FASTQ, GenBank, PDB), sequence alignment, BLAST searches, phylogenetic analysis, structure analysis, and NCBI database access.
version: 1.83
license: Biopython License
---

# Biopython - Bioinformatics Library

Industry-standard Python library for computational biology and bioinformatics workflows.

## When to Use

- Parsing and manipulating biological sequences (DNA, RNA, protein)
- Reading and writing sequence files (FASTA, FASTQ, GenBank, EMBL, SwissProt)
- Performing sequence alignments (pairwise and multiple)
- Running and parsing BLAST searches
- Analyzing protein structures from PDB files
- Calculating sequence statistics and molecular weights
- Translating DNA to protein sequences
- Finding restriction enzyme sites
- Building and analyzing phylogenetic trees
- Accessing NCBI databases (Entrez, PubMed)
- Computing sequence motifs and patterns
- Analyzing next-generation sequencing data

## Reference Documentation

**Official docs**: https://biopython.org/  
**Tutorial**: https://biopython.org/DIST/docs/tutorial/Tutorial.html  
**Search patterns**: `SeqIO.parse`, `Seq`, `AlignIO`, `NCBIWWW.qblast`, `PDBParser`

## Core Principles

### Use Biopython For

| Task | Module | Example |
|------|--------|---------|
| Create sequences | `Seq` | `Seq("ATCG")` |
| Read sequence files | `SeqIO` | `SeqIO.parse("file.fasta", "fasta")` |
| Pairwise alignment | `pairwise2` | `pairwise2.align.globalxx(s1, s2)` |
| Multiple alignment | `AlignIO` | `AlignIO.read("align.fasta", "fasta")` |
| BLAST searches | `NCBIWWW` | `NCBIWWW.qblast("blastn", "nr", seq)` |
| PDB structures | `PDB.PDBParser` | `PDBParser().get_structure()` |
| Phylogenetic trees | `Phylo` | `Phylo.read("tree.xml", "phyloxml")` |
| NCBI databases | `Entrez` | `Entrez.esearch(db="nucleotide")` |

### Do NOT Use For

- High-performance genome assembly (use SPAdes, Canu)
- Variant calling from BAM files (use GATK, BCFtools)
- RNA-seq differential expression (use DESeq2, edgeR)
- Protein structure prediction (use AlphaFold, RoseTTAFold)
- Large-scale metagenomics (use specialized pipelines)

## Quick Reference

### Installation

```bash
# pip (recommended)
pip install biopython

# With optional dependencies
pip install biopython[extra]

# conda
conda install -c conda-forge biopython

# Development version
pip install git+https://github.com/biopython/biopython.git
```

### Standard Imports

```python
# Core sequence handling
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO, AlignIO

# Sequence alignment
from Bio import pairwise2
from Bio.Align import MultipleSeqAlignment
from Bio.Align.Applications import ClustalwCommandline

# BLAST
from Bio.Blast import NCBIWWW, NCBIXML

# Structure analysis
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import PPBuilder

# Phylogenetics
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

# NCBI Entrez
from Bio import Entrez

# Additional tools
from Bio.SeqUtils import GC, molecular_weight
from Bio.Restriction import *
```

### Basic Pattern - Sequence Creation

```python
from Bio.Seq import Seq

# Create DNA sequence
dna = Seq("ATGGCCATTGTAATGGGCCGC")

# Transcribe to RNA
rna = dna.transcribe()

# Translate to protein
protein = dna.translate()

# Reverse complement
rev_comp = dna.reverse_complement()

print(f"DNA:     {dna}")
print(f"RNA:     {rna}")
print(f"Protein: {protein}")
print(f"RevComp: {rev_comp}")
```

### Basic Pattern - File Reading

```python
from Bio import SeqIO

# Read FASTA file (iterator - memory efficient)
for record in SeqIO.parse("sequences.fasta", "fasta"):
    print(f"ID: {record.id}")
    print(f"Length: {len(record.seq)}")
    print(f"Sequence: {record.seq[:50]}...")

# Read single sequence
record = SeqIO.read("single.fasta", "fasta")
```

### Basic Pattern - Sequence Alignment

```python
from Bio import pairwise2
from Bio.Seq import Seq

seq1 = Seq("ACCGT")
seq2 = Seq("ACGT")

# Global alignment
alignments = pairwise2.align.globalxx(seq1, seq2)

# Print best alignment
best = alignments[0]
print(pairwise2.format_alignment(*best))
```

## Critical Rules

### ✅ DO

- **Use iterators for large files** - `SeqIO.parse()` not `SeqIO.to_dict()`
- **Validate sequences** - Check alphabet and length before operations
- **Handle file formats correctly** - Match parser to actual file format
- **Check alignment quality** - Verify gaps and identity percentages
- **Use appropriate genetic code** - Specify table for translation
- **Close file handles** - Use context managers or explicit close
- **Filter FASTQ by quality** - Don't trust all reads equally
- **Verify PDB structure** - Check for missing atoms/residues
- **Set Entrez email** - Required for NCBI API usage
- **Handle translation frames** - Consider all three reading frames

### ❌ DON'T

- **Load entire FASTQ into memory** - Use streaming
- **Ignore sequence type** - DNA, RNA, and protein need different handling
- **Skip quality filtering** - FASTQ quality scores matter
- **Use wrong genetic code** - Different organisms use different tables
- **Forget stop codons** - Handle them explicitly in translation
- **Mix alphabets** - Don't compare DNA with protein sequences
- **Trust all BLAST hits** - Filter by e-value and identity
- **Ignore chain breaks** - PDB structures may have gaps
- **Hammer NCBI servers** - Use rate limiting (3 requests/sec without API key)
- **Compare raw sequences** - Align first, then compare

## Anti-Patterns (NEVER)

```python
# ❌ BAD: Loading entire file into memory
records = list(SeqIO.parse("huge.fastq", "fastq"))
for record in records:
    process(record)  # OOM for large files!

# ✅ GOOD: Stream processing
for record in SeqIO.parse("huge.fastq", "fastq"):
    if min(record.letter_annotations["phred_quality"]) >= 20:
        process(record)

# ❌ BAD: Translating without checking frame
protein = dna_seq.translate()
# May include stop codons or wrong frame!

# ✅ GOOD: Specify frame and stop codon handling
protein = dna_seq.translate(to_stop=True, table=1)

# ❌ BAD: No email for Entrez
Entrez.esearch(db="nucleotide", term="human")
# NCBI will block you!

# ✅ GOOD: Always set email
Entrez.email = "your.email@example.com"
handle = Entrez.esearch(db="nucleotide", term="human")

# ❌ BAD: Comparing sequences directly
if str(seq1) == str(seq2):
    print("Same")  # Ignores gaps, misalignments!

# ✅ GOOD: Align then compare
from Bio import pairwise2
alignments = pairwise2.align.globalxx(seq1, seq2)
score = alignments[0][2]  # Alignment score

# ❌ BAD: Wrong file format
records = SeqIO.parse("file.gb", "fasta")  # Wrong!

# ✅ GOOD: Match format to file
records = SeqIO.parse("file.gb", "genbank")
```

## Sequence Objects

### Creating and Manipulating Sequences

```python
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC

# DNA sequence
dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")

# Basic operations
length = len(dna)
gc_count = dna.count("G") + dna.count("C")
gc_content = (gc_count / length) * 100

# Find subsequence
position = dna.find("ATG")  # Returns index or -1

# Slicing
first_codon = dna[:3]
last_10 = dna[-10:]

print(f"Length: {length}")
print(f"GC content: {gc_content:.2f}%")
print(f"ATG at position: {position}")
```

### Transcription and Translation

```python
from Bio.Seq import Seq

# DNA to RNA transcription
dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
rna = dna.transcribe()

# RNA to protein translation
protein = rna.translate()

# DNA to protein (direct)
protein_direct = dna.translate()

# Back-transcription
dna_back = rna.back_transcribe()

# Reverse complement
rev_comp = dna.reverse_complement()

print(f"DNA:        {dna}")
print(f"RNA:        {rna}")
print(f"Protein:    {protein}")
print(f"Rev Comp:   {rev_comp}")
```

### Translation with Different Genetic Codes

```python
from Bio.Seq import Seq

dna = Seq("ATGGGCTAG")

# Standard genetic code (table 1)
protein_standard = dna.translate(table=1)

# Mitochondrial genetic code (table 2)
protein_mito = dna.translate(table=2)

# Stop at first stop codon
protein_to_stop = dna.translate(to_stop=True)

# All three reading frames
for i in range(3):
    frame = dna[i:]
    protein = frame.translate(to_stop=True)
    print(f"Frame {i}: {protein}")
```

### Sequence Utilities

```python
from Bio.Seq import Seq
from Bio.SeqUtils import GC, molecular_weight

seq = Seq("ATGGCCATTGTAATGGGCCGC")

# GC content
gc = GC(seq)

# Molecular weight
mw = molecular_weight(seq, seq_type='DNA')

# GC skew (for origin of replication analysis)
from Bio.SeqUtils import GC_skew
skew = GC_skew(seq, window=100)

print(f"GC content: {gc:.2f}%")
print(f"Molecular weight: {mw:.2f} Da")
```

## SeqRecord Objects

### Creating SeqRecord Objects

```python
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Create with minimal info
seq = Seq("ATGGCCATTG")
record = SeqRecord(seq, id="seq1", description="My sequence")

# With full annotation
record = SeqRecord(
    Seq("ATGGCCATTG"),
    id="gene1",
    name="GeneName",
    description="Gene encoding protein X",
    annotations={
        "organism": "Homo sapiens",
        "molecule_type": "DNA"
    }
)

# Add features
from Bio.SeqFeature import SeqFeature, FeatureLocation
feature = SeqFeature(
    FeatureLocation(0, 10),
    type="CDS",
    qualifiers={"product": "hypothetical protein"}
)
record.features.append(feature)

print(record)
```

### SeqRecord Attributes

```python
from Bio import SeqIO

record = SeqIO.read("sequence.gb", "genbank")

# Access components
sequence = record.seq
identifier = record.id
name = record.name
description = record.description

# Annotations (dictionary)
organism = record.annotations.get("organism", "Unknown")

# Features (list)
for feature in record.features:
    if feature.type == "CDS":
        product = feature.qualifiers.get("product", ["Unknown"])[0]
        location = feature.location
        print(f"{product} at {location}")

# Letter annotations (quality scores for FASTQ)
if "phred_quality" in record.letter_annotations:
    avg_quality = sum(record.letter_annotations["phred_quality"]) / len(record)
    print(f"Average quality: {avg_quality:.2f}")
```

## File Input/Output

### Reading FASTA Files

```python
from Bio import SeqIO

# Single sequence
record = SeqIO.read("single.fasta", "fasta")
print(f"{record.id}: {record.seq}")

# Multiple sequences (iterator)
for record in SeqIO.parse("multiple.fasta", "fasta"):
    print(f"{record.id}: {len(record.seq)} bp")

# Load into dictionary (use sparingly!)
sequences = SeqIO.to_dict(SeqIO.parse("sequences.fasta", "fasta"))
seq1 = sequences["seq1"]
```

### Writing FASTA Files

```python
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Create records
records = []
for i in range(10):
    seq = Seq("ATCG" * (i + 1))
    record = SeqRecord(seq, id=f"seq_{i}", description=f"Sequence {i}")
    records.append(record)

# Write to file
SeqIO.write(records, "output.fasta", "fasta")

# Append to existing file
with open("output.fasta", "a") as f:
    SeqIO.write(record, f, "fasta")
```

### Reading FASTQ Files

```python
from Bio import SeqIO

# Read FASTQ with quality scores
for record in SeqIO.parse("reads.fastq", "fastq"):
    seq_id = record.id
    sequence = record.seq
    quality = record.letter_annotations["phred_quality"]
    
    avg_q = sum(quality) / len(quality)
    min_q = min(quality)
    
    print(f"{seq_id}: avg Q={avg_q:.1f}, min Q={min_q}")
```

### Quality Filtering FASTQ

```python
from Bio import SeqIO

def filter_fastq(input_file, output_file, min_quality=20, min_length=50):
    """Filter FASTQ by quality and length."""
    good_reads = 0
    
    with open(output_file, "w") as out_handle:
        for record in SeqIO.parse(input_file, "fastq"):
            # Check length
            if len(record.seq) < min_length:
                continue
            
            # Check quality
            qualities = record.letter_annotations["phred_quality"]
            if min(qualities) >= min_quality:
                SeqIO.write(record, out_handle, "fastq")
                good_reads += 1
    
    return good_reads

# Filter reads
n_passed = filter_fastq("raw.fastq", "filtered.fastq")
print(f"Passed: {n_passed} reads")
```

### Reading GenBank Files

```python
from Bio import SeqIO

# GenBank files have rich annotations
for record in SeqIO.parse("sequence.gb", "genbank"):
    print(f"ID: {record.id}")
    print(f"Organism: {record.annotations['organism']}")
    print(f"Sequence length: {len(record.seq)}")
    
    # Extract features
    for feature in record.features:
        if feature.type == "CDS":
            product = feature.qualifiers.get("product", ["Unknown"])[0]
            start = feature.location.start
            end = feature.location.end
            print(f"  CDS: {product} ({start}-{end})")
```

### Converting File Formats

```python
from Bio import SeqIO

# Convert GenBank to FASTA
records = SeqIO.parse("input.gb", "genbank")
SeqIO.write(records, "output.fasta", "fasta")

# Convert FASTQ to FASTA (lose quality scores)
records = SeqIO.parse("reads.fastq", "fastq")
SeqIO.write(records, "sequences.fasta", "fasta")

# Multiple format conversions in one go
def convert_format(input_file, input_format, output_file, output_format):
    """Generic format converter."""
    count = SeqIO.convert(input_file, input_format, output_file, output_format)
    print(f"Converted {count} records")

convert_format("seq.gb", "genbank", "seq.fasta", "fasta")
```

## Sequence Alignment

### Pairwise Alignment

```python
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.pairwise2 import format_alignment

seq1 = Seq("ACCGGT")
seq2 = Seq("ACGT")

# Global alignment (Needleman-Wunsch)
alignments = pairwise2.align.globalxx(seq1, seq2)

# Print best alignment
best = alignments[0]
print(format_alignment(*best))

# With scoring matrix
alignments = pairwise2.align.globalms(
    seq1, seq2,
    match=2,      # Match score
    mismatch=-1,  # Mismatch penalty
    open=-0.5,    # Gap open penalty
    extend=-0.1   # Gap extension penalty
)
```

### Custom Scoring

```python
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.Align import substitution_matrices

seq1 = Seq("KEVLA")
seq2 = Seq("KELVA")

# Use BLOSUM62 matrix
matrix = substitution_matrices.load("BLOSUM62")

alignments = pairwise2.align.globaldx(
    seq1, seq2,
    matrix,
    open=-10,
    extend=-0.5
)

print(pairwise2.format_alignment(*alignments[0]))
```

### Local Alignment

```python
from Bio import pairwise2
from Bio.Seq import Seq

# Local alignment (Smith-Waterman)
seq1 = Seq("GCATGCTAGATGCTA")
seq2 = Seq("ATGCTA")

alignments = pairwise2.align.localxx(seq1, seq2)

# Best local alignment
best = alignments[0]
print(pairwise2.format_alignment(*best))
```

### Multiple Sequence Alignment

```python
from Bio import AlignIO
from Bio.Align.Applications import ClustalwCommandline

# Run ClustalW (requires installation)
cline = ClustalwCommandline("clustalw2", infile="sequences.fasta")
stdout, stderr = cline()

# Read alignment
alignment = AlignIO.read("sequences.aln", "clustal")

print(f"Alignment length: {alignment.get_alignment_length()}")
print(f"Number of sequences: {len(alignment)}")

# Print alignment
print(alignment)

# Access sequences
for record in alignment:
    print(f"{record.id}: {record.seq}")
```

### Alignment Statistics

```python
from Bio import AlignIO

alignment = AlignIO.read("alignment.fasta", "fasta")

# Calculate identity
def calculate_identity(alignment):
    """Calculate pairwise identity matrix."""
    n = len(alignment)
    identities = []
    
    for i in range(n):
        for j in range(i+1, n):
            seq1 = alignment[i].seq
            seq2 = alignment[j].seq
            
            matches = sum(a == b for a, b in zip(seq1, seq2) if a != '-' and b != '-')
            aligned_length = sum(1 for a, b in zip(seq1, seq2) if a != '-' and b != '-')
            
            identity = matches / aligned_length * 100 if aligned_length > 0 else 0
            identities.append(identity)
    
    return identities

identities = calculate_identity(alignment)
print(f"Average identity: {sum(identities)/len(identities):.2f}%")
```

## BLAST Searches

### Running BLAST Online

```python
from Bio.Blast import NCBIWWW, NCBIXML
from Bio import SeqIO

# Read query sequence
record = SeqIO.read("query.fasta", "fasta")

# Run BLAST
result_handle = NCBIWWW.qblast(
    program="blastn",     # blastn, blastp, blastx, tblastn, tblastx
    database="nt",        # nt, nr, refseq_rna, etc.
    sequence=str(record.seq),
    hitlist_size=10
)

# Save results
with open("blast_results.xml", "w") as out_handle:
    out_handle.write(result_handle.read())

result_handle.close()
```

### Parsing BLAST Results

```python
from Bio.Blast import NCBIXML

# Parse BLAST XML output
with open("blast_results.xml") as result_handle:
    blast_records = NCBIXML.parse(result_handle)
    
    for blast_record in blast_records:
        print(f"Query: {blast_record.query}")
        print(f"Number of hits: {len(blast_record.alignments)}")
        
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                if hsp.expect < 0.001:  # E-value threshold
                    print(f"\n  Hit: {alignment.title}")
                    print(f"  Length: {alignment.length}")
                    print(f"  E-value: {hsp.expect}")
                    print(f"  Identity: {hsp.identities}/{hsp.align_length} "
                          f"({hsp.identities/hsp.align_length*100:.1f}%)")
                    print(f"  Query:   {hsp.query}")
                    print(f"  Subject: {hsp.sbjct}")
```

### BLAST with Custom Parameters

```python
from Bio.Blast import NCBIWWW

result_handle = NCBIWWW.qblast(
    program="blastp",
    database="nr",
    sequence=protein_seq,
    expect=0.001,          # E-value threshold
    word_size=3,           # Word size
    matrix_name="BLOSUM62", # Substitution matrix
    gapcosts="11 1",       # Gap costs (open, extend)
    hitlist_size=50,       # Number of hits
    filter="L"             # Low complexity filter
)
```

### Batch BLAST

```python
from Bio.Blast import NCBIWWW, NCBIXML
from Bio import SeqIO
import time

def batch_blast(fasta_file, output_file):
    """Run BLAST for multiple sequences."""
    results = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        print(f"BLASTing {record.id}...")
        
        result_handle = NCBIWWW.qblast(
            "blastn", "nt",
            str(record.seq),
            hitlist_size=5
        )
        
        blast_record = NCBIXML.read(result_handle)
        results.append((record.id, blast_record))
        
        # Be nice to NCBI servers
        time.sleep(1)
    
    return results

# Run batch BLAST
results = batch_blast("queries.fasta", "batch_results.xml")
```

## Protein Structure Analysis

### Loading PDB Files

```python
from Bio.PDB import PDBParser

# Create parser
parser = PDBParser(QUIET=True)

# Load structure
structure = parser.get_structure("protein", "protein.pdb")

# Access hierarchy: Structure → Model → Chain → Residue → Atom
model = structure[0]
chain = model['A']

print(f"Structure ID: {structure.id}")
print(f"Number of models: {len(structure)}")
print(f"Number of chains: {len(model)}")
print(f"Number of residues in chain A: {len(chain)}")
```

### Extracting Information

```python
from Bio.PDB import PDBParser

parser = PDBParser()
structure = parser.get_structure("protein", "protein.pdb")

# Iterate through structure
for model in structure:
    for chain in model:
        print(f"Chain {chain.id}")
        
        for residue in chain:
            # Skip hetero atoms (water, ligands)
            if residue.id[0] == ' ':
                resname = residue.resname
                resid = residue.id[1]
                
                # Get atoms
                for atom in residue:
                    atom_name = atom.name
                    coord = atom.coord
                    print(f"  {resname}{resid} {atom_name}: {coord}")
```

### Calculating Distances

```python
from Bio.PDB import PDBParser
import numpy as np

parser = PDBParser()
structure = parser.get_structure("protein", "protein.pdb")

# Get two atoms
chain = structure[0]['A']
atom1 = chain[10]['CA']  # CA atom of residue 10
atom2 = chain[20]['CA']  # CA atom of residue 20

# Calculate distance
distance = atom1 - atom2  # Overloaded operator
print(f"Distance: {distance:.2f} Å")

# Manual calculation
coord1 = atom1.coord
coord2 = atom2.coord
dist = np.linalg.norm(coord1 - coord2)
print(f"Distance (manual): {dist:.2f} Å")
```

### Center of Mass

```python
from Bio.PDB import PDBParser
import numpy as np

parser = PDBParser()
structure = parser.get_structure("protein", "protein.pdb")

def calculate_center_of_mass(entity):
    """Calculate center of mass for Structure/Model/Chain."""
    coords = []
    masses = []
    
    for atom in entity.get_atoms():
        coords.append(atom.coord)
        masses.append(atom.mass)
    
    coords = np.array(coords)
    masses = np.array(masses)
    
    com = np.average(coords, axis=0, weights=masses)
    return com

# Calculate for whole structure
com = calculate_center_of_mass(structure)
print(f"Center of mass: {com}")
```

### Selecting Atoms

```python
from Bio.PDB import PDBParser, Selection

parser = PDBParser()
structure = parser.get_structure("protein", "protein.pdb")

# Select all atoms
all_atoms = Selection.unfold_entities(structure, 'A')
print(f"Total atoms: {len(all_atoms)}")

# Select specific atoms
model = structure[0]

# All CA atoms
ca_atoms = [atom for atom in model.get_atoms() if atom.name == 'CA']
print(f"CA atoms: {len(ca_atoms)}")

# Specific residue range
chain_a = model['A']
residues_10_to_20 = [chain_a[i] for i in range(10, 21) if i in chain_a]
```

### Secondary Structure with DSSP

```python
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

parser = PDBParser()
structure = parser.get_structure("protein", "protein.pdb")
model = structure[0]

# Run DSSP (requires dssp executable)
dssp = DSSP(model, "protein.pdb")

# Extract secondary structure
for key in dssp:
    residue = dssp[key]
    chain_id = key[0]
    res_id = key[1][1]
    ss = residue[2]  # Secondary structure: H=helix, E=sheet, C=coil
    acc = residue[3]  # Accessible surface area
    
    print(f"{chain_id} {res_id}: {ss} (ASA={acc:.1f})")
```

### Extracting Sequence from PDB

```python
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

parser = PDBParser()
structure = parser.get_structure("protein", "protein.pdb")

# Build polypeptides
ppb = PPBuilder()

for chain in structure[0]:
    for pp in ppb.build_peptides(chain):
        sequence = pp.get_sequence()
        print(f"Chain {chain.id}: {sequence}")
```

### Writing PDB Files

```python
from Bio.PDB import PDBParser, PDBIO, Select

# Load structure
parser = PDBParser()
structure = parser.get_structure("protein", "input.pdb")

# Select what to save
class ChainSelect(Select):
    def accept_chain(self, chain):
        return chain.id == 'A'  # Only save chain A

# Write structure
io = PDBIO()
io.set_structure(structure)
io.save("chain_A.pdb", ChainSelect())
```

## Phylogenetic Analysis

### Building Trees from Alignment

```python
from Bio import AlignIO, Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

# Read alignment
alignment = AlignIO.read("alignment.fasta", "fasta")

# Calculate distance matrix
calculator = DistanceCalculator('identity')
dm = calculator.get_distance(alignment)

# Construct tree (Neighbor-Joining)
constructor = DistanceTreeConstructor(calculator, 'nj')
tree = constructor.build_tree(alignment)

# Save tree
Phylo.write(tree, "tree.xml", "phyloxml")

print(tree)
```

### Tree Visualization

```python
from Bio import Phylo

# Load tree
tree = Phylo.read("tree.xml", "phyloxml")

# Draw tree (ASCII)
Phylo.draw_ascii(tree)

# Draw tree (graphical - requires matplotlib)
import matplotlib.pyplot as plt
Phylo.draw(tree)
plt.savefig("tree.png")

# Interactive view
# tree.ladderize()  # Flip tree for better visualization
# Phylo.draw(tree, do_show=True)
```

### Tree Analysis

```python
from Bio import Phylo

tree = Phylo.read("tree.xml", "phyloxml")

# Get all terminals (leaves)
terminals = tree.get_terminals()
print(f"Number of leaves: {len(terminals)}")

# Get all non-terminals (internal nodes)
nonterminals = tree.get_nonterminals()
print(f"Number of internal nodes: {len(nonterminals)}")

# Find common ancestor
clade1 = tree.find_any(name="Species1")
clade2 = tree.find_any(name="Species2")
common = tree.common_ancestor(clade1, clade2)

# Calculate distances
dist = tree.distance(clade1, clade2)
print(f"Distance between Species1 and Species2: {dist:.4f}")

# Get path between nodes
path = tree.get_path(clade1, clade2)
```

### Creating Trees Programmatically

```python
from Bio.Phylo.BaseTree import Tree, Clade

# Create tree structure
tree = Tree()

# Create clades (nodes)
clade_a = Clade(branch_length=0.5, name="A")
clade_b = Clade(branch_length=0.3, name="B")
clade_c = Clade(branch_length=0.4, name="C")

# Create internal node
internal = Clade(branch_length=0.2)
internal.clades = [clade_a, clade_b]

# Create root
root = Clade()
root.clades = [internal, clade_c]

tree.root = root

# Visualize
Phylo.draw_ascii(tree)
```

## NCBI Entrez

### Searching Databases

```python
from Bio import Entrez

# ALWAYS set your email
Entrez.email = "your.email@example.com"

# Search nucleotide database
handle = Entrez.esearch(
    db="nucleotide",
    term="Homo sapiens[Organism] AND COX1",
    retmax=10
)

record = Entrez.read(handle)
handle.close()

print(f"Found {record['Count']} records")
print(f"IDs: {record['IdList']}")
```

### Fetching Records

```python
from Bio import Entrez, SeqIO

Entrez.email = "your.email@example.com"

# Fetch by ID
handle = Entrez.efetch(
    db="nucleotide",
    id="NC_000001",
    rettype="gb",
    retmode="text"
)

record = SeqIO.read(handle, "genbank")
handle.close()

print(f"ID: {record.id}")
print(f"Description: {record.description}")
print(f"Length: {len(record.seq)}")
```

### Batch Downloads

```python
from Bio import Entrez, SeqIO

Entrez.email = "your.email@example.com"

def download_sequences(id_list, output_file):
    """Download multiple sequences by ID."""
    # Fetch records
    handle = Entrez.efetch(
        db="nucleotide",
        id=id_list,
        rettype="fasta",
        retmode="text"
    )
    
    records = SeqIO.parse(handle, "fasta")
    SeqIO.write(records, output_file, "fasta")
    handle.close()

# Download
ids = ["NM_001301717", "NM_001301718", "NM_001301719"]
download_sequences(ids, "downloaded.fasta")
```

### Entrez with History

```python
from Bio import Entrez

Entrez.email = "your.email@example.com"

# Search with history
search_handle = Entrez.esearch(
    db="nucleotide",
    term="human[organism] AND COX1",
    usehistory="y",
    retmax=1000
)

search_results = Entrez.read(search_handle)
search_handle.close()

webenv = search_results["WebEnv"]
query_key = search_results["QueryKey"]
count = int(search_results["Count"])

print(f"Found {count} records")

# Fetch in batches
batch_size = 100
for start in range(0, count, batch_size):
    fetch_handle = Entrez.efetch(
        db="nucleotide",
        rettype="fasta",
        retmode="text",
        retstart=start,
        retmax=batch_size,
        webenv=webenv,
        query_key=query_key
    )
    
    # Process batch
    data = fetch_handle.read()
    fetch_handle.close()
```

## Advanced Workflows

### Complete Gene Analysis Pipeline

```python
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import GC

Entrez.email = "your.email@example.com"

def analyze_gene(gene_id):
    """Complete analysis of a gene."""
    # 1. Fetch sequence
    handle = Entrez.efetch(db="nucleotide", id=gene_id, rettype="gb", retmode="text")
    record = SeqIO.read(handle, "genbank")
    handle.close()
    
    # 2. Basic stats
    length = len(record.seq)
    gc_content = GC(record.seq)
    
    # 3. Extract CDS
    cds_list = []
    for feature in record.features:
        if feature.type == "CDS":
            cds_seq = feature.extract(record.seq)
            product = feature.qualifiers.get("product", ["Unknown"])[0]
            cds_list.append((product, cds_seq))
    
    # 4. Translate CDS
    proteins = []
    for product, cds in cds_list:
        try:
            protein = cds.translate(to_stop=True)
            proteins.append((product, protein))
        except:
            proteins.append((product, None))
    
    return {
        'id': record.id,
        'length': length,
        'gc': gc_content,
        'n_cds': len(cds_list),
        'proteins': proteins
    }

# Analyze gene
result = analyze_gene("NM_000518")
print(f"Gene: {result['id']}")
print(f"Length: {result['length']} bp")
print(f"GC: {result['gc']:.2f}%")
print(f"CDS: {result['n_cds']}")
```

### RNA-Seq Read Processing

```python
from Bio import SeqIO
import numpy as np

def process_rnaseq_reads(fastq_file, min_quality=30, min_length=50):
    """Process RNA-seq reads with quality filtering."""
    stats = {
        'total': 0,
        'passed_quality': 0,
        'passed_length': 0,
        'final': 0
    }
    
    passed_reads = []
    
    for record in SeqIO.parse(fastq_file, "fastq"):
        stats['total'] += 1
        
        # Quality filter
        qualities = record.letter_annotations["phred_quality"]
        if np.mean(qualities) < min_quality:
            continue
        stats['passed_quality'] += 1
        
        # Length filter
        if len(record.seq) < min_length:
            continue
        stats['passed_length'] += 1
        
        # Trim adapters (simple example)
        # In reality, use specialized tools like Cutadapt
        
        passed_reads.append(record)
        stats['final'] += 1
    
    return passed_reads, stats

# Process reads
reads, stats = process_rnaseq_reads("rnaseq.fastq")
print(f"Total: {stats['total']}")
print(f"Passed quality: {stats['passed_quality']}")
print(f"Passed length: {stats['passed_length']}")
print(f"Final: {stats['final']}")
```

### Restriction Enzyme Analysis

```python
from Bio import SeqIO
from Bio.Restriction import *

# Load sequence
record = SeqIO.read("plasmid.gb", "genbank")
seq = record.seq

# Create restriction batch
rb = RestrictionBatch([EcoRI, BamHI, PstI, HindIII])

# Find restriction sites
analysis = rb.search(seq)

# Print results
for enzyme, sites in analysis.items():
    if sites:
        print(f"{enzyme}: {len(sites)} site(s) at {sites}")
    else:
        print(f"{enzyme}: No sites found")

# Find enzymes that cut once (good for cloning)
single_cutters = [enz for enz, sites in analysis.items() if len(sites) == 1]
print(f"\nSingle cutters: {[str(e) for e in single_cutters]}")
```

### Codon Usage Analysis

```python
from Bio import SeqIO
from collections import Counter

def analyze_codon_usage(genbank_file):
    """Analyze codon usage in CDS features."""
    codon_counts = Counter()
    total_codons = 0
    
    for record in SeqIO.parse(genbank_file, "genbank"):
        for feature in record.features:
            if feature.type == "CDS":
                # Extract CDS sequence
                cds = feature.extract(record.seq)
                
                # Count codons
                for i in range(0, len(cds) - 2, 3):
                    codon = str(cds[i:i+3])
                    if len(codon) == 3:
                        codon_counts[codon] += 1
                        total_codons += 1
    
    # Calculate frequencies
    codon_freq = {codon: count/total_codons 
                  for codon, count in codon_counts.items()}
    
    return codon_counts, codon_freq

# Analyze
counts, freqs = analyze_codon_usage("genome.gb")

# Print most common codons
for codon, freq in sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{codon}: {freq:.4f}")
```

## Performance Optimization

### Memory-Efficient File Processing

```python
from Bio import SeqIO

def process_large_fasta(filename, process_func):
    """Process large FASTA without loading into memory."""
    count = 0
    
    for record in SeqIO.parse(filename, "fasta"):
        process_func(record)
        count += 1
        
        # Progress report
        if count % 10000 == 0:
            print(f"Processed {count} sequences")
    
    return count

# Example processor
def calculate_gc(record):
    from Bio.SeqUtils import GC
    gc = GC(record.seq)
    if gc > 60:
        print(f"{record.id}: High GC ({gc:.1f}%)")

# Process
n = process_large_fasta("large_genome.fasta", calculate_gc)
```

### Parallel Processing

```python
from Bio import SeqIO
from multiprocessing import Pool
from functools import partial

def process_sequence(record, min_length=100):
    """Process single sequence."""
    if len(record.seq) >= min_length:
        from Bio.SeqUtils import GC
        return (record.id, len(record.seq), GC(record.seq))
    return None

def parallel_process_fasta(filename, n_processes=4):
    """Process FASTA in parallel."""
    # Load sequences
    records = list(SeqIO.parse(filename, "fasta"))
    
    # Process in parallel
    with Pool(n_processes) as pool:
        results = pool.map(process_sequence, records)
    
    # Filter None results
    return [r for r in results if r is not None]

# Process
results = parallel_process_fasta("sequences.fasta", n_processes=4)
for seq_id, length, gc in results[:10]:
    print(f"{seq_id}: {length} bp, GC={gc:.2f}%")
```

### Index Files for Random Access

```python
from Bio import SeqIO

# Create index (fast random access)
record_dict = SeqIO.index("large.fasta", "fasta")

# Access specific sequence instantly
record = record_dict["seq_12345"]
print(record.seq)

# Iterate (still memory efficient)
for key in record_dict:
    record = record_dict[key]
    process(record)

# Close when done
record_dict.close()

# SQLite-backed index (for very large files)
record_dict = SeqIO.index_db("large.idx", "huge.fasta", "fasta")
```

## Common Pitfalls and Solutions

### Translation Frame Errors

```python
from Bio.Seq import Seq

dna = Seq("ATGGCCATTGTAATGGGCCGC")

# Problem: Wrong reading frame
protein = dna.translate()  # May have stop codons

# Solution: Check all frames
for i in range(3):
    frame = dna[i:]
    protein = frame.translate(to_stop=True, table=1)
    if len(protein) > 10:  # Reasonable length
        print(f"Frame {i}: {protein}")
```

### File Format Mismatches

```python
from Bio import SeqIO

# Problem: Wrong format specified
try:
    records = SeqIO.parse("file.fasta", "genbank")  # Wrong!
except:
    print("Format mismatch")

# Solution: Verify format
import os

def guess_format(filename):
    ext = os.path.splitext(filename)[1].lower()
    format_map = {
        '.fasta': 'fasta',
        '.fa': 'fasta',
        '.fna': 'fasta',
        '.fastq': 'fastq',
        '.fq': 'fastq',
        '.gb': 'genbank',
        '.gbk': 'genbank'
    }
    return format_map.get(ext, 'fasta')

format = guess_format("file.fasta")
records = SeqIO.parse("file.fasta", format)
```

### NCBI Rate Limiting

```python
from Bio import Entrez
import time

Entrez.email = "your.email@example.com"

# Problem: Too many requests
def bad_batch_download(id_list):
    for gene_id in id_list:
        handle = Entrez.efetch(db="nucleotide", id=gene_id)
        # This will get you blocked!

# Solution: Add delays
def good_batch_download(id_list):
    results = []
    for gene_id in id_list:
        handle = Entrez.efetch(db="nucleotide", id=gene_id, 
                               rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        results.append(record)
        
        time.sleep(0.34)  # ~3 requests/sec
    return results

# Better: Use Entrez history for large batches
```

### Memory Issues with Large Alignments

```python
from Bio import AlignIO

# Problem: Loading huge alignment
alignment = AlignIO.read("huge_alignment.fasta", "fasta")  # OOM!

# Solution: Process incrementally
def process_alignment_chunks(filename, chunk_size=1000):
    records = SeqIO.parse(filename, "fasta")
    
    chunk = []
    for record in records:
        chunk.append(record)
        
        if len(chunk) >= chunk_size:
            # Process chunk
            process_chunk(chunk)
            chunk = []
    
    # Process remaining
    if chunk:
        process_chunk(chunk)
```

This comprehensive Biopython guide covers 50+ examples across all major bioinformatics workflows!
