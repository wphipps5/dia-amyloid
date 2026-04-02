NSAF Composite Index Calculation
Purpose
nsaf-composite-index-calc.py computes quantitative amyloid typing indices from protein‑level normalized spectral abundance factor (NSAF) data derived from data‑dependent acquisition (DDA) proteomics analyses.

The script implements a continuous scoring framework for several amyloid types by combining absolute protein abundance with relative abundance among biologically related proteins. Hill‑type transformations are used to calibrate bounded composite indices that maximize discrimination between a target amyloid type and other amyloid types in the dataset.

Indices are calculated at the replicate level, aggregated to the case level, and visualized using type‑versus‑other plots.

Script
nsaf-composite-index-calc.py

Required Inputs
The script expects the following files in the working directory.

spectral-counts.txt
Tab‑delimited table containing normalized spectral abundance factor (NSAF) values for all detected proteins across LC–MS/MS analyses.

protein-type.txt
Table defining groups of proteins associated with each amyloid type used for index calculations. This file maps amyloid type codes to UniProt accessions.

Example Directory Structure
The script assumes the input files are located in the same working directory.

Example:

nsaf_index_calculation/

nsaf-composite-index-calc.py
spectral-counts.txt
protein-type.txt

Output
All results are written to an output/ directory.

Final organized outputs include:

output/index_calc_by_repl/
Replicate‑level index calculations and supporting tables

output/index_calc_by_case/
Case‑level index summaries and visualizations

output/index_calc_by_case/tables/
Case‑level index tables

output/index_calc_by_case/plots/
Case‑level diagnostic plots

Intermediate staging directories used during computation are removed automatically at the end of the workflow.

Indices Calculated
The script generates composite indices for several amyloid types including:

Lambda light chain amyloidosis
Kappa light chain amyloidosis
Fibrinogen A amyloidosis
Keratin‑associated amyloidosis

In addition, a lambda–kappa differential index is calculated to support discrimination between AL lambda and AL kappa amyloidosis.

Run
From the directory containing the input files:

python nsaf-composite-index-calc.py

Notes
Index parameters are optimized using differential evolution to maximize type‑versus‑other discrimination based on AUROC.

Composite indices combine:

absolute abundance of the target protein
relative abundance compared to related proteins
bounded Hill‑type transformations
Replicate‑level indices are averaged to produce case‑level values, which are then used to generate diagnostic plots and summary tables.
