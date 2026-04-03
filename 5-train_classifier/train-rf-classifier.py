#!/usr/bin/env python3
"""
train_rf_classifier.py

Train and evaluate a Random Forest classifier for amyloid typing
using DIA peptide features.

Required input (working directory):
    rf_training_data.csv

Output directory (created automatically):
    output/

Outputs:
    per_fold_performance.csv
    summary_performance.csv
    per_fold_classification_report.csv
    classification_report.csv
    confusion_matrix.csv
    feature_importance.csv
    rf_model.joblib
    feature_schema.csv
    label_encoder.joblib
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

# Input training dataset
INPUT_FILE = "rf_training_data.csv"

# Output directory
OUT_DIR = "output"

# ------------------------------------------------------------
# Cross‑validation settings
# ------------------------------------------------------------

# Number of stratified CV folds
N_SPLITS = 5

# Random seed for reproducibility
RANDOM_STATE = 42

# ------------------------------------------------------------
# Random Forest model parameters
# ------------------------------------------------------------

# Number of trees in the forest
N_TREES = 500

# Class weighting strategy
CLASS_WEIGHT = "balanced"

# Number of parallel CPU threads (-1 = all cores)
N_JOBS = -1

# ------------------------------------------------------------
# Initialize output directory
# ------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n[STEP 1] Loading training data")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

if "Type" not in df.columns:
    raise RuntimeError("rf_training_data.csv must contain 'Type' column")

feature_df = df.drop(columns=["Type"])
X = feature_df.values
feature_names = feature_df.columns.tolist()
y = df["Type"].values

print(f"[INFO] Samples: {X.shape[0]}")
print(f"[INFO] Features: {X.shape[1]}")

# ------------------------------------------------------------
# ENCODE LABELS
# ------------------------------------------------------------

le = LabelEncoder()
y_enc = le.fit_transform(y)
class_labels = le.classes_

print(f"[INFO] Classes: {list(class_labels)}")

# ------------------------------------------------------------
# CROSS‑VALIDATION
# ------------------------------------------------------------

print("\n[STEP 2] Cross‑validation")

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

pooled_preds = []
pooled_true = []

per_fold_metrics = []
per_fold_reports = []
feature_importances = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_enc), start=1):

    print(f"[CV] Fold {fold_idx}/{N_SPLITS}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT,
        n_jobs=N_JOBS
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # scalar metrics
    per_fold_metrics.append({
        "Fold": fold_idx,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="macro", zero_division=0)
    })

    # per-class report
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_labels,
        output_dict=True,
        zero_division=0
    )

    df_report = pd.DataFrame(report).transpose()
    df_report["Fold"] = fold_idx
    df_report["Class"] = df_report.index

    df_report = df_report[
        ["Fold", "Class", "precision", "recall", "f1-score", "support"]
    ]

    per_fold_reports.append(df_report.reset_index(drop=True))

    pooled_preds.extend(y_pred)
    pooled_true.extend(y_test)

    feature_importances.append(clf.feature_importances_)

# ------------------------------------------------------------
# SAVE CROSS‑VALIDATION RESULTS
# ------------------------------------------------------------

print("\n[STEP 3] Saving cross‑validation results")

df_perf = pd.DataFrame(per_fold_metrics)

df_perf.to_csv(
    os.path.join(OUT_DIR, "per_fold_performance.csv"),
    index=False
)

summary_rows = []

for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
    summary_rows.append({
        "Metric": metric,
        "Mean": df_perf[metric].mean(),
        "SD": df_perf[metric].std()
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    os.path.join(OUT_DIR, "summary_performance.csv"),
    index=False
)

pd.concat(per_fold_reports, ignore_index=True).to_csv(
    os.path.join(OUT_DIR, "per_fold_classification_report.csv"),
    index=False
)

# ------------------------------------------------------------
# POOLED PERFORMANCE
# ------------------------------------------------------------

print("[STEP 4] Computing pooled metrics")

pooled_report = classification_report(
    pooled_true,
    pooled_preds,
    target_names=class_labels,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(pooled_report).transpose().to_csv(
    os.path.join(OUT_DIR, "classification_report.csv")
)

cm = confusion_matrix(pooled_true, pooled_preds)

pd.DataFrame(
    cm,
    index=class_labels,
    columns=class_labels
).to_csv(
    os.path.join(OUT_DIR, "confusion_matrix.csv")
)

# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

print("[STEP 5] Computing feature importance")

mean_importance = np.mean(feature_importances, axis=0)

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": mean_importance
}).sort_values("Importance", ascending=False)

fi_df.to_csv(
    os.path.join(OUT_DIR, "feature_importance.csv"),
    index=False
)

# ------------------------------------------------------------
# TRAIN FINAL MODEL
# ------------------------------------------------------------
print("[STEP 6] Training final model")

# Create model package directory
MODEL_DIR = os.path.join(OUT_DIR, "model_package")
os.makedirs(MODEL_DIR, exist_ok=True)

final_model = RandomForestClassifier(
    n_estimators=N_TREES,
    random_state=RANDOM_STATE,
    class_weight=CLASS_WEIGHT,
    n_jobs=N_JOBS
)

final_model.fit(X, y_enc)

# Save trained model
joblib.dump(
    final_model,
    os.path.join(MODEL_DIR, "rf_model.joblib")
)

# Save feature schema
pd.DataFrame({"Feature": feature_names}).to_csv(
    os.path.join(MODEL_DIR, "feature_schema.csv"),
    index=False
)

# Save label encoder
joblib.dump(
    le,
    os.path.join(MODEL_DIR, "label_encoder.joblib")
)

print(f"[OK] Model package written to: {MODEL_DIR}")
print("\n[DONE] Random Forest training complete")