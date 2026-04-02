# dia-amyloid
DIA Amyloid Proteomics Pipeline
This repository contains scripts used for proteomic analysis and classification of amyloid deposits from tissue biopsies using liquid chromatography–tandem mass spectrometry (LC–MS/MS). The workflow integrates data‑dependent acquisition (DDA) proteomics with data‑independent acquisition (DIA) peptide quantification and machine learning–based classification.

The scripts implement a reproducible pipeline for assembling proteomic search databases, extracting diagnostic peptide features, training classification models, and computing quantitative indices used in amyloid typing.

Overview
The pipeline combines protein‑level and peptide‑level proteomic analysis to support amyloid subtype classification. Initial stages use DDA proteomics to identify proteins and derive quantitative indices based on normalized spectral abundance factors (NSAF). These data are then used to define an empirical amyloid peptidome, which forms the basis for DIA peptide quantification and machine‑learning classification.

Pipeline Structure
The repository is organized into numbered directories corresponding to the major stages of the analysis workflow.

1‑dda_fasta_assembly
Construction of the proteomic search FASTA database including canonical human proteins, isoforms, missense variants, and supplemental amyloid‑associated sequences.

2‑dda_nsaf_composite_index_calc
Computation of protein‑level normalized spectral abundance factor (NSAF) indices and diagnostic visualizations from DDA proteomics data.

3‑dda_peptidome_extraction
Extraction of an empirical amyloid peptidome from DDA peptide identifications.

4‑Process_DIA_Peptides
Processing of DIA peptide measurements and construction of the DIA feature matrix.

5‑Train_Classifier
Training and evaluation of machine‑learning models for amyloid subtype classification.

6‑dia_test_set_evaluation
Application of trained classifiers to independent or prospective DIA datasets.

The numbered directory structure reflects the order in which scripts are typically executed during analysis.
