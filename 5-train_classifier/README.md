# Random Forest Classifier Training

## Purpose

`train_rf_classifier.py` trains and evaluates a Random Forest classifier for amyloid subtype classification using peptide‑level DIA features.

The script performs stratified cross‑validation, calculates performance metrics, evaluates feature importance, and trains a final classifier using the full dataset. The trained model and supporting metadata are saved for downstream inference on prospective samples.

---

## Script

train_rf_classifier.py

---

## Required Inputs

The script expects the following file in the working directory.

rf_training_data.csv

Tab‑delimited dataset containing peptide‑level DIA feature measurements for each LC–MS/MS replicate.

The dataset must contain:

- a column named **Type** representing the amyloid subtype label
- additional columns representing peptide features (quantitative DIA measurements)

Example structure:

ReplicateFeature1Feature2Feature3...Type

Each row represents a single DIA replicate used for classifier training.

---

## Example Directory Structure

5-Train_Classifier/

train_rf_classifier.py  
rf_training_data.csv  
README.md

---

## Output

All results are written to the directory:

output/

### Cross‑Validation Performance

per_fold_performance.csv  
Performance metrics for each cross‑validation fold.

summary_performance.csv  
Mean and standard deviation of accuracy, precision, recall, and F1‑score across folds.

per_fold_classification_report.csv  
Per‑class performance metrics for each fold.

classification_report.csv  
Combined classification report across pooled predictions.

confusion_matrix.csv  
Confusion matrix for pooled predictions across all folds.

---

### Feature Analysis

feature_importance.csv

Table ranking peptide features by mean importance across Random Forest models.

This provides insight into which peptides contribute most strongly to amyloid subtype classification.

---

### Trained Model Package

output/model_package/

rf_model.joblib  
Serialized trained Random Forest model.

feature_schema.csv  
List of feature columns expected by the model.

label_encoder.joblib  
Encoder mapping subtype labels to numeric class indices.

This package can be used for applying the classifier to new DIA datasets.

---

## Model Configuration

Random Forest parameters:

N_TREES = 500  
Number of decision trees used in the ensemble.

CLASS_WEIGHT = balanced  
Adjusts class weights to compensate for class imbalance.

N_JOBS = -1  
Uses all available CPU cores for training.

---

## Cross‑Validation

The model is evaluated using stratified K‑fold cross‑validation:

N_SPLITS = 5

Each fold preserves the distribution of amyloid subtypes across training and validation sets.

Performance metrics include:

- Accuracy  
- Precision (macro‑averaged)  
- Recall (macro‑averaged)  
- F1 score (macro‑averaged)

---

## Run

Execute the script from the directory containing the training dataset:

python train_rf_classifier.py

---

## Notes

The script first evaluates classifier performance using cross‑validation before training the final model on the complete dataset.

Feature importance values represent the average importance of each peptide feature across cross‑validation folds.

The resulting model package is used by downstream scripts for amyloid subtype prediction on prospective DIA datasets.
