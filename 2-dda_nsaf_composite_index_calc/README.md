**NSAF Composite Index Calculation**

**Purpose**

nsaf-composite-index-calc.py computes quantitative amyloid typing indices from protein-level normalized spectral abundance factor (NSAF) data derived from data-dependent acquisition (DDA) proteomics. The script calibrates composite indices using bounded Hill-type transformations that combine: **(1)** absolute abundance of amyloid precursor proteins **(2)** relative abundance relationships among biologically related proteins.

Indices are computed at the replicate level, aggregated to the case level, and visualized using Type-versus-Other diagnostic plots. These continuous scores support conservative interpretation of amyloid subtype classification.

**Script**
nsaf-composite-index-calc.py

**Required Inputs**
The script expects the following files in the working directory:

**spectral-counts.txt**

Tab-delimited table containing normalized spectral abundance factor (NSAF) values for proteins across LC-MS/MS analyses.

Typical structure:
ProteinID Description Sample1 Sample2 Sample3 ...

Values represent normalized spectral abundance factors derived from DDA proteomics.

protein-type.txt
Table defining groups of proteins associated with each amyloid type used for index calculations.

This file maps amyloid type codes to UniProt accessions.

Example structure:

Type Protein1 Protein2 Protein3 ...
ALL P0DOY3 P0DOY2
ALK P0DOY2 P0DOY3
FIB P02671
KER P13647 P02533

These mappings define which proteins contribute to each composite index.

audit-codes.csv (optional)
Optional annotation file specifying replicate-level audit classifications used for downstream grouping and visualization.

Required columns:

CodeName,Audit-Code

If this file is not present, all replicates are assumed to be technically valid.

Example Directory Structure
2-dda_nsaf_composite_index_calc/

nsaf-composite-index-calc.py
spectral-counts.txt
protein-type.txt
audit-codes.csv
README.md

Output
All results are written to an output/ directory.

Final organized outputs include:

output/index_calc_by_repl/

Replicate-level index calculations and supporting tables.

output/index_calc_by_case/

Case-level index summaries and visualizations.

output/index_calc_by_case/tables/

CASE-level index tables.

output/index_calc_by_case/plots/

Diagnostic plots showing Type-versus-Other distributions.

Intermediate staging directories used during computation are automatically removed after processing.

Indices Calculated
The script computes composite indices for several amyloid types including:

Lambda light chain amyloidosis
Kappa light chain amyloidosis
Fibrinogen A amyloidosis
Keratin-associated amyloidosis

A lambda-kappa differential index is also calculated to support discrimination between AL lambda and AL kappa amyloidosis.

Run
Execute the script from the directory containing the input files:

python nsaf-composite-index-calc.py

Notes
Composite indices are calibrated using differential evolution optimization to maximize discrimination between the target amyloid type and other amyloid types in the dataset.

Each index integrates:

absolute NSAF signal of the target protein
relative enrichment compared to biologically related proteins
bounded Hill-type transformations to stabilize dynamic range
Replicate-level indices are averaged to produce case-level scores used for diagnostic visualization.
