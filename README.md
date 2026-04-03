# DIA Amyloid Proteomics Pipeline

This repository contains scripts used for proteomic analysis and classification of amyloid deposits from tissue biopsies using liquid chromatography–tandem mass spectrometry (LC–MS/MS).

The workflow integrates:

- data‑dependent acquisition (DDA) proteomics  
- data‑independent acquisition (DIA) peptide quantification  
- machine‑learning classification of amyloid subtype  

The pipeline implements a reproducible framework for assembling proteomic search databases, extracting diagnostic peptide features, training classification models, and evaluating prospective samples.

---

## Conceptual Workflow

The analysis progresses from protein‑level proteomics to peptide‑level machine learning classification.

Tissue biopsy proteomics  
Data‑dependent acquisition (DDA) → Protein identification and NSAF quantification → Amyloid proteome (dominant precursor protein signals)  
→ Empirical amyloid peptidome (peptides derived from abundant proteins) → Data‑independent acquisition (DIA) peptide quantification → Peptide feature matrix → Random Forest classification → Amyloid subtype prediction

This framework mirrors the analytical strategy described in the manuscript, where protein‑level enrichment patterns identified by DDA proteomics are translated into peptide‑level quantitative features measured by DIA.

---

# Repository Structure

The repository is organized into numbered directories corresponding to the major stages of the pipeline.

## 1-dda_fasta_assembly

Construction of the proteomic search FASTA database including:

- canonical human proteins  
- isoforms  
- missense variants  
- supplemental amyloid‑associated sequences  

Script:

amyloid-dda-fasta.py

---

## 2-dda_nsaf_composite_index_calc

Computation of protein‑level normalized spectral abundance factor (NSAF) indices and diagnostic visualizations from DDA proteomics data.

These indices integrate:

- absolute precursor protein abundance  
- proportional relationships among related proteins  

Script:

nsaf-composite-index-calc.py

---

## 3-dda_peptidome_extraction

Extraction of an empirical **amyloid peptidome** from DDA peptide identification results.

Steps include:

- ranking proteins by NSAF  
- selecting dominant proteins  
- extracting peptide evidence from Percolator output  
- generating peptide lists for downstream DIA analysis  

Script:

extract-peptidome.py

---

## 4-process_dia_peptides

Processing of peptide measurements from DIA LC–MS/MS analyses exported from Skyline.

Functions include:

- quality control filtering  
- fragment intensity normalization  
- construction of the DIA peptide feature matrix  
- AUROC evaluation of diagnostic peptide signals  

Script:

process-dia-peptides.py

---

## 5-train_classifier

Training and evaluation of Random Forest classifiers for amyloid subtype classification.

Outputs include:

- classifier performance metrics  
- feature importance analysis  
- trained model package  

Script:

train_rf_classifier.py

---

## 6-dia_test_set_evaluation

Application of trained classifiers to prospective DIA datasets.

The script:

- constructs peptide feature matrices for new samples  
- generates subtype probability predictions  
- computes peptide evidence scores for low‑confidence predictions  

Script:

infer_prospective_cases.py

---

# Software Requirements

Python 3 is required.

Required packages:

numpy  
pandas  
scikit‑learn  
scipy  
matplotlib  
joblib  

Install dependencies using:

pip install -r requirements.txt


---

# Data Inputs

Several pipeline steps require user‑supplied proteomics datasets.

## DDA data

- spectral counts derived from DDA database search results  
- Percolator peptide output files  

These files are dataset‑specific and are not included in this repository.

## DIA data

Skyline export reports containing peptide fragment ion measurements.

Typical columns include:

Replicate Name  
Protein  
Precursor  
Total Area Fragment  
Total Area  
Isotope Dot Product  
Library Dot Product  

---

# Typical Workflow

1. Build the DDA search FASTA database  
2. Identify proteins and peptides from DDA LC–MS/MS data  
3. Compute NSAF‑based composite indices  
4. Extract the empirical amyloid peptidome  
5. Process DIA peptide measurements  
6. Train the Random Forest classifier  
7. Apply the classifier to prospective samples  

---

# Notes

The workflow reflects interpretive strategies commonly used in proteomic amyloid typing, combining enrichment of amyloid precursor proteins with relative abundance relationships among related proteins.

Machine‑learning classification is performed using peptide‑level features derived from DIA analyses.
