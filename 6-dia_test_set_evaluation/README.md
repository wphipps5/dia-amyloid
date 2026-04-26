# DIA Amyloid Type Classifier

## Purpose

amyloid-type-classifier.py applies a trained Random Forest classifier to prospective DIA samples and computes peptide‑level evidence scores for low‑confidence predictions.

The script:

- constructs a normalized peptide feature matrix from Skyline DIA export reports
- aligns the feature matrix to the training schema used during model development
- applies the trained classifier to generate predicted amyloid types and class probability estimates
- computes peptide evidence scores for each sample using AUROC‑derived signal thresholds

---

## Script

amyloid-type-classifier.py

---

## Required Inputs

The script expects the following files and directories in the working directory.

### Skyline-DIA-Reports/

Directory containing Skyline export reports for prospective LC–DIA–MS/MS analyses.

Expected filename pattern:

DIA-report_*.csv

Each report must contain the following columns:

- Replicate Name
- Protein
- Precursor
- Total Area Fragment
- Total Area
- Isotope Dot Product
- Library Dot Product

---

### model_package/

Directory containing the trained model package produced by train-rf-classifier.py.

Required files:

rf_model.joblib
Trained Random Forest classifier.

feature_schema.csv
Ordered list of peptide features used during training.

label_encoder.joblib
Label encoder mapping class labels to numeric identifiers.

---

### peptide_thresholds/

Directory containing AUROC‑based peptide audit tables produced by process-dia-peptides.py.

Expected files:

Audit_<TYPE>.csv
One file per amyloid type, containing BestCutoff and SpecificityCutoff thresholds for each peptide feature.

---

## Example Directory Structure

6-dia_test_set_evaluation/

amyloid-type-classifier.py
model_package/
    rf_model.joblib
    feature_schema.csv
    label_encoder.joblib
peptide_thresholds/
    Audit_THY.csv
    Audit_ALL.csv
    Audit_ALK.csv
    ...
Skyline-DIA-Reports/
    DIA-report_batch1.csv
    DIA-report_batch2.csv
README.md

---

## Output

All results are written to:

output/

---

### prospective_dia_feature_matrix.csv

Normalized peptide feature matrix for prospective samples, formatted to match the training data structure.

Columns include Replicate, Type (parsed from replicate name), and peptide feature columns.

---

### rf_predictions.csv

Predicted amyloid type and classifier confidence for each sample.

Columns include:

- CodeName: replicate identifier
- Predicted_Type: predicted amyloid subtype
- Confidence: maximum class probability
- P_<TYPE>: per‑class probability for each amyloid type
- Low_Confidence: flag indicating predictions below the confidence threshold (default 0.50)

---

### <TYPE>-feature-score-matrix.csv

Peptide evidence score matrix for each amyloid type. One file is produced per amyloid type based on available Audit_<TYPE>.csv files.

Each peptide feature is scored on a three‑point scale per sample:

1 point: peptide signal detected above zero
1 point: signal exceeds the AUROC‑derived discrimination threshold (BestCutoff)
1 point: signal exceeds the specificity threshold (SpecificityCutoff)

Scores are summed across all peptides belonging to each amyloid type.

---

### composite-low-confidence-scores.csv

Summary table of total peptide evidence scores across all amyloid types for each sample.

---

## Quality Control Filtering

Peptide measurements are filtered using the same thresholds applied during training.

Initial filtering

DOTP_THRESHOLD = 0.7

Strict filtering

ISO_MIN = 0.9
LIB_MIN = 0.8

---

## Key Parameters

CONFIDENCE_THRESHOLD = 0.5
Predictions below this threshold are flagged in rf_predictions.csv and assessed using peptide evidence scores.

MODEL_DIR = model_package
Directory containing the trained model package.

AUDIT_DIR = peptide_thresholds
Directory containing peptide threshold tables.

---

## Run

python amyloid-type-classifier.py

---

## Notes

The peptide feature matrix is aligned to the training feature schema before prediction. Features present in the training data but absent in the prospective dataset are filled with zero.

Peptide evidence scoring provides a complementary assessment of amyloid type signal strength independent of the Random Forest probability estimates. It is particularly useful for evaluating samples with ambiguous or low‑confidence classifier outputs.
