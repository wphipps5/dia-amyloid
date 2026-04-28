#!/usr/bin/env python3
"""
train_rf_classifier.py

Train and evaluate a Random Forest classifier for amyloid typing
using DIA peptide features.

Required input (working directory):
    rf_training_data.csv

Outputs (created automatically):
    output/block_naive/
        per_fold_performance.csv
        summary_performance.csv
        per_fold_classification_report.csv
        classification_report.csv
        confusion_matrix.csv
        feature_importance.csv
        model_package/
            rf_model.joblib
            feature_schema.csv
            label_encoder.joblib

    output/block_grouped/
        per_fold_performance.csv
        summary_performance.csv
        per_fold_classification_report.csv
        classification_report.csv
        confusion_matrix.csv
        feature_importance.csv
"""

import os
import re
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
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

INPUT_FILE   = "rf_training_data.csv"
OUT_NAIVE    = os.path.join("output", "block_naive")
OUT_GROUPED  = os.path.join("output", "block_grouped")
MODEL_DIR    = os.path.join(OUT_NAIVE, "model_package")

N_SPLITS     = 5
RANDOM_STATE = 42
N_TREES      = 500
CLASS_WEIGHT = "balanced"
N_JOBS       = -1

for d in [OUT_NAIVE, OUT_GROUPED, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("\n[STEP 1] Loading training data")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

if "Type" not in df.columns:
    raise RuntimeError("rf_training_data.csv must contain 'Type' column")

rep_ids      = df["Replicate"].values
feature_df   = df.drop(columns=["Type", "Replicate"])
X            = feature_df.values
feature_names = feature_df.columns.tolist()
y            = df["Type"].values

print(f"[INFO] Samples:  {X.shape[0]}")
print(f"[INFO] Features: {X.shape[1]}")

le           = LabelEncoder()
y_enc        = le.fit_transform(y)
class_labels = le.classes_

print(f"[INFO] Classes: {list(class_labels)}")

# ============================================================
# HELPER: run one CV loop and write outputs to out_dir
# ============================================================

def run_cv(splitter, X, y_enc, groups, out_dir, label):

    pooled_preds      = []
    pooled_true       = []
    per_fold_metrics  = []
    per_fold_reports  = []
    feature_imps      = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(X, y_enc, groups), start=1
    ):
        print(f"[{label}] Fold {fold_idx}/{N_SPLITS}")

        clf = RandomForestClassifier(
            n_estimators=N_TREES,
            random_state=RANDOM_STATE,
            class_weight=CLASS_WEIGHT,
            n_jobs=N_JOBS
        )
        clf.fit(X[train_idx], y_enc[train_idx])
        y_pred = clf.predict(X[test_idx])
        y_test = y_enc[test_idx]

        per_fold_metrics.append({
            "Fold":      fold_idx,
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall":    recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1-score":  f1_score(y_test, y_pred, average="macro", zero_division=0)
        })

        report = classification_report(
            y_test, y_pred,
            target_names=class_labels,
            output_dict=True,
            zero_division=0
        )
        df_rep = pd.DataFrame(report).transpose()
        df_rep["Fold"]  = fold_idx
        df_rep["Class"] = df_rep.index
        df_rep = df_rep[["Fold", "Class", "precision", "recall", "f1-score", "support"]]
        per_fold_reports.append(df_rep.reset_index(drop=True))

        pooled_preds.extend(y_pred)
        pooled_true.extend(y_test)
        feature_imps.append(clf.feature_importances_)

    # per-fold performance
    df_perf = pd.DataFrame(per_fold_metrics)
    df_perf.to_csv(os.path.join(out_dir, "per_fold_performance.csv"), index=False)

    # summary
    pd.DataFrame([
        {"Metric": m, "Mean": df_perf[m].mean(), "SD": df_perf[m].std()}
        for m in ["Accuracy", "Precision", "Recall", "F1-score"]
    ]).to_csv(os.path.join(out_dir, "summary_performance.csv"), index=False)

    # per-fold classification report
    pd.concat(per_fold_reports, ignore_index=True).to_csv(
        os.path.join(out_dir, "per_fold_classification_report.csv"), index=False
    )

    # pooled classification report
    pooled_report = classification_report(
        pooled_true, pooled_preds,
        target_names=class_labels,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(pooled_report).transpose().to_csv(
        os.path.join(out_dir, "classification_report.csv")
    )

    # confusion matrix
    cm = confusion_matrix(pooled_true, pooled_preds)
    pd.DataFrame(cm, index=class_labels, columns=class_labels).to_csv(
        os.path.join(out_dir, "confusion_matrix.csv")
    )

    # feature importance
    pd.DataFrame({
        "Feature":    feature_names,
        "Importance": np.mean(feature_imps, axis=0)
    }).sort_values("Importance", ascending=False).to_csv(
        os.path.join(out_dir, "feature_importance.csv"), index=False
    )

    print(f"[OK] {label} results written to: {out_dir}")


# ============================================================
# STEP 2 — BLOCK-NAIVE CV (StratifiedKFold at injection level)
# ============================================================

print("\n[STEP 2] Block-naive cross-validation (injection level)")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
run_cv(skf, X, y_enc, groups=None, out_dir=OUT_NAIVE, label="NAIVE CV")

# ============================================================
# STEP 3 — BLOCK-GROUPED CV (StratifiedGroupKFold at block level)
# ============================================================

print("\n[STEP 3] Block-grouped cross-validation (tissue block level)")

def extract_block_id(rep_name):
    m = re.search(r'DIA-([A-Z]{3}\d{4})', str(rep_name))
    return m.group(1) if m else str(rep_name)

groups = np.array([extract_block_id(r) for r in rep_ids])
print(f"[INFO] Unique tissue blocks: {len(set(groups))}")

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS)
run_cv(sgkf, X, y_enc, groups=groups, out_dir=OUT_GROUPED, label="GROUPED CV")

# ============================================================
# STEP 4 — TRAIN FINAL MODEL (on full dataset)
# ============================================================

print("\n[STEP 4] Training final model on full dataset")

final_model = RandomForestClassifier(
    n_estimators=N_TREES,
    random_state=RANDOM_STATE,
    class_weight=CLASS_WEIGHT,
    n_jobs=N_JOBS
)
final_model.fit(X, y_enc)

joblib.dump(final_model, os.path.join(MODEL_DIR, "rf_model.joblib"))
pd.DataFrame({"Feature": feature_names}).to_csv(
    os.path.join(MODEL_DIR, "feature_schema.csv"), index=False
)
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

print(f"[OK] Model package written to: {MODEL_DIR}")
print("\n[DONE] Random Forest training complete")
