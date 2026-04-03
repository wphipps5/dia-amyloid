# Random Forest Classifier Training

## Purpose

`train_rf_classifier.py` trains and evaluates a Random Forest classifier for amyloid subtype classification using peptide‑level DIA features.

The script performs stratified cross‑validation to estimate classification performance, computes feature importance scores, and trains a final classifier using the full dataset. The trained model and associated metadata are saved for downstream prediction of amyloid subtype in prospective DIA samples.

---

## Script

train_rf_classifier.py

---

## Required Inputs

### rf_training_data.csv

Training dataset containing peptide‑level DIA measurements used as input features for the classifier.

Each row represents a single LC–MS/MS replicate.

The dataset must contain:

• a column named **Type** indicating the amyloid subtype label  
• multiple columns representing quantitative peptide features

Example structure:

Type,feature1,feature2,feature3,...,featureN

Example:

Type,ap|P02766-x|K.TSSEGLHGLTEEEEFVGVK.V|624,ap|P02766-x|R.GSPAINVAVHFR.K|612,...
ALL,0.0000132,0,0.0000081,...
ALK,0,0.0000215,0,...
THY,0.0000187,0,0,...

Feature columns correspond to peptide precursor features extracted from DIA analyses. Values represent normalized peptide fragment ion intensities.

Important requirements:

• The **Type column must exist** and contain amyloid subtype labels.  
• All other columns are treated as numeric peptide features.  
• Missing values should be represented as **0**.  
• Each row corresponds to **one DIA replicate**.

---

## Example Directory Structure

5-Train_Classifier/

train_rf_classifier.py  
rf_training_data.csv  
README.md

---

## Output

All results are written to:

output/

### Cross‑Validation Performance

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

---

### Feature Analysis

feature_importance.csv

Table ranking peptide features by mean importance across cross‑validation models. This provides insight into which peptide signals contribute most strongly to subtype discrimination.

---

### Trained Model Package

output/model_package/

rf_model.joblib  
Serialized Random Forest classifier.

feature_schema.csv  
List of feature columns used during training.

label_encoder.joblib  
Encoder mapping subtype labels to numeric class indices.

These files are required for applying the classifier to prospective datasets.

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

The classifier is evaluated using stratified K‑fold cross‑validation.

N_SPLITS = 5

Stratification ensures that each fold preserves the distribution of amyloid subtypes.

Performance metrics include:

• Accuracy  
• Precision (macro‑averaged)  
• Recall (macro‑averaged)  
• F1 score (macro‑averaged)

---

## Run

Run the script from the directory containing the training dataset:

python train_rf_classifier.py

---

## Notes

The script first evaluates classifier performance using cross‑validation before training a final model on the complete dataset.

Feature importance values represent the average contribution of each peptide feature across the cross‑validation models.

The resulting model package is used by downstream scripts to predict amyloid subtype for prospective DIA datasets.
