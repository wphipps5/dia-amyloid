# Random Forest Classifier Training

## Purpose

train-rf-classifier.py trains and evaluates a Random Forest classifier for amyloid subtype classification using peptide‑level DIA features.

The script performs two stratified cross‑validation strategies to estimate classification performance, computes feature importance scores, and trains a final classifier using the full dataset. The trained model and associated metadata are saved for downstream prediction of amyloid subtype in prospective DIA samples.

---

## Script

train-rf-classifier.py

---

## Required Inputs

### rf_training_data.csv

Training dataset containing peptide‑level DIA measurements used as input features for the classifier.

Each row represents a single LC–MS/MS replicate.

The dataset must contain:

• a column named **Replicate** containing the LC–MS/MS injection identifier
• a column named **Type** indicating the amyloid subtype label
• multiple columns representing quantitative peptide features

Example structure:

Replicate,Type,feature1,feature2,feature3,...,featureN

Example:

DIA-THY0218-240101-A.1,THY,0.0000187,0,0,...
DIA-ALL0234-240227-A.1,ALL,0.0000132,0,0.0000081,...
DIA-ALK0157-240227-A.1,ALK,0,0.0000215,0,...

Feature columns correspond to peptide precursor features extracted from DIA analyses. Values represent normalized peptide fragment ion intensities. Missing values should be represented as 0.

Important requirements:

• The **Replicate** column must exist and contain injection identifiers used for tissue block extraction during block‑grouped cross‑validation.
• The **Type** column must exist and contain amyloid subtype labels.
• All other columns are treated as numeric peptide features.

---

## Example Directory Structure

5-train_classifier/

train-rf-classifier.py
rf_training_data.csv
README.md

---

## Output

All results are written to subdirectories of:

output/

Two sets of cross‑validation outputs are produced, plus a trained model package.

---

### output/block_naive/

Results from injection‑level cross‑validation (StratifiedKFold).

per_fold_performance.csv
Performance metrics for each cross‑validation fold.

summary_performance.csv
Mean and standard deviation of accuracy, precision, recall, and F1‑score across folds.

per_fold_classification_report.csv
Per‑class performance metrics for each fold.

classification_report.csv
Combined classification report based on pooled predictions.

confusion_matrix.csv
Confusion matrix summarizing classifier performance.

feature_importance.csv
Table ranking peptide features by mean importance across cross‑validation models.

---

### output/block_grouped/

Results from tissue block‑level cross‑validation (StratifiedGroupKFold).

Contains the same set of output files as output/block_naive/.

---

### output/block_naive/model_package/

Trained model package saved after fitting the final classifier on the full dataset.

rf_model.joblib
Serialized Random Forest classifier trained on the full dataset.

feature_schema.csv
Ordered list of peptide features used during training.

label_encoder.joblib
Encoder mapping subtype labels to numeric class indices.

These files are required for applying the classifier to prospective datasets using amyloid-type-classifier.py.

---

## Model Configuration

Default Random Forest parameters:

N_TREES = 500
Number of decision trees in the ensemble.

CLASS_WEIGHT = balanced
Adjusts class weights to compensate for class imbalance.

N_JOBS = -1
Uses all available CPU cores during training.

---

## Cross‑Validation

The classifier is evaluated using two stratified five‑fold cross‑validation strategies applied to the same training data.

### Injection‑level cross‑validation (StratifiedKFold)

N_SPLITS = 5

Individual LC–MS/MS injections are assigned to folds independently. Stratification preserves the distribution of amyloid subtypes across training and test partitions. Results are written to output/block_naive/.

### Block‑level cross‑validation (StratifiedGroupKFold)

N_SPLITS = 5

Tissue block identifiers are used as grouping variables, ensuring that all injections derived from the same tissue block are assigned exclusively to either the training or test fold. Block identifiers are extracted from replicate names using the pattern DIA‑[TYPE][BLOCK]. This strategy prevents data leakage across injections from the same specimen. Results are written to output/block_grouped/.

Performance metrics include:

• Accuracy
• Precision (macro‑averaged)
• Recall (macro‑averaged)
• F1‑score (macro‑averaged)

---

## Run

Run the script from the directory containing the training dataset:

python train-rf-classifier.py

---

## Notes

The script first evaluates classifier performance using both cross‑validation strategies before training a final model on the complete dataset.

Feature importance values represent the average contribution of each peptide feature across the injection‑level cross‑validation models.

The resulting model package is used by amyloid-type-classifier.py to predict amyloid subtype for prospective DIA datasets.
