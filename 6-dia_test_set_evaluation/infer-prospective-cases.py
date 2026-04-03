#!/usr/bin/env python3
"""
infer_prospective_cases.py
Apply trained RF classifier to prospective DIA samples
and compute peptide evidence scores.
Implements pipeline Steps 26–28.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

REPORT_DIR = "Skyline-DIA-Reports"

# Model package directory produced by train_rf_classifier.py
MODEL_DIR = "model_package"
MODEL_FILE = os.path.join(MODEL_DIR, "rf_model.joblib")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.joblib")
SCHEMA_FILE = os.path.join(MODEL_DIR, "feature_schema.csv")

# Directory containing Audit_<TYPE>.csv threshold tables
AUDIT_DIR = "peptide_thresholds"

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# Skyline QC thresholds
DOTP_THRESHOLD = 0.7
ISO_MIN = 0.9
LIB_MIN = 0.8

# RF confidence cutoff for triggering peptide scoring
CONFIDENCE_THRESHOLD = 0.5


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def extract_accession(feature):
    m = re.search(r"\|([A-Z0-9]+)-x\|", feature)
    return m.group(1) if m else ""

def extract_type_from_codename(cn):
    if not isinstance(cn, str) or len(cn) < 7:
        return ""
    return cn[4:7].upper()

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

model = joblib.load(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)

schema = pd.read_csv(SCHEMA_FILE)
training_features = schema["Feature"].tolist()


# ------------------------------------------------------------
# PROCESS PROSPECTIVE REPORTS
# ------------------------------------------------------------

rows = []

files = sorted(
    f for f in os.listdir(REPORT_DIR)
    if f.startswith("DIA-report_") and f.endswith(".csv")
)

for fname in files:

    df = pd.read_csv(os.path.join(REPORT_DIR, fname))

    cols = {c.lower(): c for c in df.columns}

    rep = cols["replicate name"]
    prot = cols["protein"]
    prec = cols["precursor"]
    frag = cols["total area fragment"]
    iso = cols["isotope dot product"]
    lib = cols["library dot product"]
    area = cols["total area"]

    df = df[(df[iso] >= DOTP_THRESHOLD) & (df[lib] >= DOTP_THRESHOLD)]

    df = df[~df[prot].str.lower().str.contains("decoy")]

    psnf = df.groupby(rep)[area].sum().to_dict()

    df = df[
        (df[iso] >= ISO_MIN) &
        (df[lib] >= LIB_MIN) &
        (df[frag] > 0)
    ]

    df["Feature"] = df[prot].astype(str) + "-" + df[prec].astype(str)

    df["MS2"] = df[frag] / df[rep].map(psnf)

    for _, r in df.iterrows():
        rows.append({
            "CodeName": r[rep],
            "Feature": r["Feature"],
            "MS2": r["MS2"]
        })


# ------------------------------------------------------------
# BUILD FEATURE MATRIX
# ------------------------------------------------------------

df_feat = pd.DataFrame(rows)

df_feat = (
    df_feat
    .groupby(["CodeName", "Feature"], as_index=False)
    .agg({"MS2": "max"})
)

matrix = df_feat.pivot(
    index="CodeName",
    columns="Feature",
    values="MS2"
).fillna(0)

# align to training schema
matrix = matrix.reindex(columns=training_features, fill_value=0)

# ------------------------------------------------------------
# EXPORT FEATURE MATRIX (TRAINING FORMAT)
# ------------------------------------------------------------

matrix_export = matrix.copy()

matrix_export.insert(
    0,
    "Type",
    [extract_type_from_codename(cn) for cn in matrix_export.index]
)

matrix_export.insert(
    0,
    "Replicate",
    matrix_export.index
)

matrix_export.to_csv(
    os.path.join(OUT_DIR, "prospective_dia_feature_matrix.csv"),
    index=False
)

print("Prospective feature matrix written")

# ------------------------------------------------------------
# APPLY RF MODEL
# ------------------------------------------------------------

X = matrix.values

probs = model.predict_proba(X)

pred_idx = np.argmax(probs, axis=1)

preds = encoder.inverse_transform(pred_idx)

conf = probs[np.arange(len(preds)), pred_idx]

df_pred = pd.DataFrame({
    "CodeName": matrix.index,
    "Predicted_Type": preds,
    "Confidence": conf
})

for i, cls in enumerate(encoder.classes_):
    df_pred[f"P_{cls}"] = probs[:, i]

df_pred["Low_Confidence"] = df_pred["Confidence"] < CONFIDENCE_THRESHOLD

df_pred.to_csv(
    os.path.join(OUT_DIR, "rf_predictions.csv"),
    index=False
)

print("RF predictions written")


# ------------------------------------------------------------
# PEPTIDE EVIDENCE SCORING
# ------------------------------------------------------------

audit_files = [
    f for f in os.listdir(AUDIT_DIR)
    if f.startswith("Audit_")
]

scores = {}

for f in audit_files:

    type_code = f.replace("Audit_", "").replace(".csv", "")

    df_a = pd.read_csv(os.path.join(AUDIT_DIR, f))

    df_a = df_a[df_a["BelongsToType"] == True]

    best = dict(zip(df_a["Feature"], df_a["BestCutoff"]))
    spec = dict(zip(df_a["Feature"], df_a["SpecificityCutoff"]))

    mat = pd.DataFrame(index=matrix.index)

    for feat in best:

        vals = matrix.get(feat, pd.Series(0, index=matrix.index))

        detected = (vals > 0).astype(int)
        above_best = (vals >= best[feat]).astype(int)
        above_spec = (vals >= spec[feat]).astype(int)

        mat[feat] = detected + above_best + above_spec

    mat["Score"] = mat.sum(axis=1)

    mat.to_csv(
        os.path.join(OUT_DIR, f"{type_code}-feature-score-matrix.csv")
    )

    scores[type_code] = mat["Score"]


# ------------------------------------------------------------
# COMPOSITE TABLE
# ------------------------------------------------------------

df_comp = pd.DataFrame(scores)

df_comp.to_csv(
    os.path.join(OUT_DIR, "composite-low-confidence-scores.csv")
)

print("Composite peptide score table written")