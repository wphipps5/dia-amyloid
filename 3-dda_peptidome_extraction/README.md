# DDA Peptidome Extraction

## Purpose

`extract-peptidome.py` identifies dominant proteins from normalized spectral abundance factor (NSAF) data and extracts peptide evidence for those proteins from Percolator peptide output files.

The script constructs an empirical **amyloid peptidome** by identifying peptides associated with the most abundant proteins in the DDA proteomics dataset. These peptides are later used as diagnostic features for downstream DIA peptide quantification and classification.

The workflow performs the following steps:

- rank proteins by maximum NSAF across all samples
- select the top N proteins (default: 500)
- extract corresponding UniProt accessions
- scan Percolator peptide output files
- retain peptides mapping to the selected proteins
- aggregate peptide evidence across samples
- export peptide tables and Skyline-compatible peptide lists

---

## Script

`extract-peptidome.py`

---

## Required Inputs

The script expects the following files and directories in the working directory.

### spectral-counts.txt

Tab-delimited table containing normalized spectral abundance factor (NSAF) values for proteins across LC–MS/MS analyses.

Typical structure:

ProteinID    Description    Sample1    Sample2    Sample3 ...

The first column must contain protein identifiers (typically FASTA headers).  
Columns after the second column should contain numeric NSAF values.

---

### percolator_peptides/

Directory containing Percolator peptide output files.

Expected filename pattern:

*_peptides.txt

Each file should contain peptide-spectrum match results including:

- peptide sequence
- protein assignment
- q-value
- flanking amino acids

These files are typically generated from Percolator post-processing of database search results.

---

## Example Directory Structure

3-dda_peptidome_extraction/

extract-peptidome.py  
spectral-counts.txt  
percolator_peptides/  
sample1_peptides.txt  
sample2_peptides.txt  
sample3_peptides.txt  
README.md

---

## Output

All results are written to the directory:

extracted-peptidome/

Key outputs include:

**top-proteins.tsv**  
Ranked list of proteins based on maximum NSAF values across replicates.

**top-protein-ids.txt**  
List of UniProt accessions for the selected top proteins.

**top-protein-peptides.tsv**  
Complete table of peptide evidence associated with the selected proteins.

**top-protein-peptides-unique.tsv**  
Non-redundant list of unique peptide sequences observed for the selected proteins.

**amyloid-peptidome.txt**  
Peptide list formatted for import into Skyline for DIA chromatogram extraction.

---

## Key Parameters

Default parameters defined in the script:

TOP_PROTEIN_COUNT = 500  
Number of proteins selected based on maximum NSAF values.

QVALUE_THRESHOLD = 0.01  
Percolator q-value threshold for peptide filtering.

---

## Run
python extract-peptidome.py


---

## Notes

Protein identifiers are parsed from FASTA-style headers to extract UniProt accessions.

Peptides are filtered using Percolator q-values and restricted to proteins ranked among the top NSAF signals in the dataset.

The resulting peptide list represents an empirical **amyloid peptidome** derived from DDA proteomic observations. This peptide set forms the basis for downstream DIA peptide quantification and machine-learning classification workflows.
