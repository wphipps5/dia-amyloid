# DIA Peptide Feature Processing

## Purpose

This script processes peptide-level quantitative data extracted from data-independent acquisition (DIA) mass spectrometry analyses exported from Skyline. It performs quality control filtering, constructs a peptide feature matrix, evaluates peptide-level diagnostic performance, and prepares datasets used for downstream machine-learning classification of amyloid subtype.

The script transforms raw DIA chromatogram measurements into a structured peptide feature space that can be used for statistical evaluation and classifier training.

Major functions include:

- filtering Skyline peptide measurements using dot-product quality metrics  
- removing decoy identifications  
- normalizing peptide fragment intensities using run-specific signal normalization  
- constructing a peptide feature matrix for each LC–MS/MS replicate  
- auditing peptide diagnostic performance using AUROC analysis  
- generating peptide threshold tables for subtype discrimination  
- producing a machine-learning training dataset for amyloid subtype classification

---

## Script

process-dia-peptides.py

---

## Required Inputs

The script expects the following files and directories in the working directory.

### Skyline-DIA-Reports/

Directory containing Skyline export reports for each LC–DIA–MS/MS analysis.

Each report should contain peptide fragment ion measurements and associated quality metrics including:

- Replicate Name  
- Protein  
- Precursor  
- Total Area Fragment  
- Total Area  
- Isotope Dot Product  
- Library Dot Product  

Expected filename pattern:

DIA-report_*.csv

These files are typically generated using Skyline report templates configured for DIA peptide quantification.

---

### protein-type.txt

Tab-delimited table mapping amyloid subtype codes to UniProt accessions of associated precursor proteins.

This file is used to determine which peptides belong to each amyloid type during diagnostic feature evaluation.

Example structure:

Type    Protein1    Protein2    Protein3 ...

ALL     P0DOY3      P0DOY2  
ALK     P0DOY2      P0DOY3  
FIB     P02671  
KER     P13647      P02533  

---

### protein_ids.txt

List of UniProt accessions representing amyloid-associated proteins used to restrict the peptide feature space.

Only peptide features derived from these proteins are retained during feature filtering.

---

## Example Directory Structure

4-Process_DIA_Peptides/

process-dia-peptides.py  
protein-type.txt  
protein_ids.txt  
Skyline-DIA-Reports/  
DIA-report_sample1.csv  
DIA-report_sample2.csv  
DIA-report_sample3.csv  
README.md  

---

## Output

All results are written to the directory:

output/

Major outputs include:

### dia_feature_matrix.csv

Peptide feature matrix containing normalized fragment ion intensities for each replicate.

Columns include Replicate (injection identifier), Type (amyloid subtype parsed from replicate name), and peptide precursor features. Rows correspond to LC–MS/MS replicates.

---

### rf_training_data.csv

Machine-learning training dataset derived from the peptide feature matrix.

Columns include Replicate (injection identifier), Type (amyloid subtype label), and peptide features restricted to amyloid-associated proteins. Rows are filtered to amyloid types supported by the classifier.

---

### peptide_thresholds/

Directory containing peptide diagnostic performance tables derived from AUROC analysis.

Files include:

Audit_<TYPE>.csv  
Diagnostic performance metrics for peptides associated with each amyloid type.

audit_compiled.csv  
Compiled table of diagnostic peptides across all amyloid types.

audit_compiled_top3.csv  
Top three diagnostic peptides per amyloid type ranked by sensitivity and specificity.

---

## Quality Control Filtering

Peptide measurements are filtered using several quality control criteria:

Initial filtering

DOTP_THRESHOLD = 0.7

Strict filtering

ISO_MIN = 0.9  
LIB_MIN = 0.8

Additional filtering steps include:

- removal of decoy protein entries  
- removal of zero fragment signals  
- replicate-level signal normalization using summed total area (PSNF)

---

## Feature Selection

Two levels of feature filtering are applied:

1. Restrict peptide features to proteins listed in protein_ids.txt  
2. Restrict machine-learning features to peptides belonging to known amyloid precursor proteins

These steps reduce noise and limit the feature space to biologically relevant peptide signals.

---

## Diagnostic Peptide Evaluation

Peptide features are evaluated for subtype discrimination using receiver operating characteristic (ROC) analysis.

For each peptide:

- AUROC is calculated  
- optimal classification threshold is estimated using the Youden index  
- sensitivity and specificity are calculated  
- peptide subtype assignment is determined from protein mappings

Diagnostic peptides are compiled into subtype-specific threshold tables for downstream interpretation.

---

## Run

Execute the script from the directory containing the input files:

python process-dia-peptides.py

---

## Notes

Peptide fragment ion intensities are normalized within each LC–MS/MS run to account for variation in total signal intensity.

The resulting peptide feature matrix provides the quantitative input used for downstream machine-learning classification of amyloid subtype.
