#!/usr/bin/env python3

import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# ============================================================
# CONFIGURATION
# ============================================================

# Input directories
INPUT_DIR = "Skyline-DIA-Reports"

# Reference files
PROTEIN_TYPE_FILE = "protein-type.txt"
PROTEIN_IDS_FILE = "protein_ids.txt"

# Output directory
OUT_DIR = "output"

# ------------------------------------------------------------
# Quality control thresholds
# ------------------------------------------------------------

# Initial Skyline dot-product filtering
DOTP_THRESHOLD = 0.7

# Strict QC thresholds
ISO_MIN = 0.9
LIB_MIN = 0.8

# ------------------------------------------------------------
# Sample filtering
# ------------------------------------------------------------

# Amyloid types excluded from all analyses
EXCLUDE_TYPES = {"INA", "NON", "UNK"}

# Amyloid types included in RF classifier training
ML_TYPES = {"THY", "LT2", "ALL", "ALK", "SAA"}

# ------------------------------------------------------------
# Feature filtering
# ------------------------------------------------------------

# Restrict peptide features to proteins listed in protein_ids.txt
RESTRICT_TO_PROTEIN_LIST = True

# Restrict RF features to peptides from amyloid proteins
RESTRICT_RF_TO_AMYLOID_PROTEINS = True

# ------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

def extract_type_from_codename(cn):
    if not isinstance(cn, str) or len(cn) < 7:
        return ""
    return cn[4:7].upper()


def extract_accession(feature):
    m = re.search(r"\|([A-Z0-9]+)-x\|", str(feature))
    return m.group(1) if m else ""


def load_type_map(path):
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    type_map = {}
    for _, row in df.iterrows():
        t = row.iloc[0].strip().upper()
        proteins = {
            str(v).strip().upper()
            for v in row.iloc[1:]
            if str(v).strip()
        }
        type_map[t] = proteins
    return type_map


print("[STEP 1] Loading Skyline reports")

files = sorted(
    f for f in os.listdir(INPUT_DIR)
    if f.startswith("DIA-report_") and f.endswith(".csv")
)

if not files:
    raise RuntimeError("No DIA reports found")

reduced_rows = []

for i, fname in enumerate(files, start=1):
    print(f"[STEP 1] Reading file {i}/{len(files)}: {fname}")

    path = os.path.join(INPUT_DIR, fname)
    df = pd.read_csv(path, low_memory=False)

    cols = {c.lower(): c for c in df.columns}

    rep = cols["replicate name"]
    prot = cols["protein"]
    prec = cols["precursor"]
    frag = cols["total area fragment"]
    iso = cols["isotope dot product"]
    lib = cols["library dot product"]
    area = cols["total area"]

    df[iso] = pd.to_numeric(df[iso], errors="coerce")
    df[lib] = pd.to_numeric(df[lib], errors="coerce")
    df[frag] = pd.to_numeric(df[frag], errors="coerce")
    df[area] = pd.to_numeric(df[area], errors="coerce")

    # Step 18: dot-product filtering
    df = df[(df[iso] >= DOTP_THRESHOLD) & (df[lib] >= DOTP_THRESHOLD)]

    # remove decoys
    df = df[~df[prot].str.lower().str.contains("decoy")]

    # PSNF calculation
    psnf = df.groupby(rep)[area].sum().to_dict()

    # Step 19: strict QC
    df = df[
        (df[iso] >= ISO_MIN) &
        (df[lib] >= LIB_MIN) &
        (df[frag] > 0)
    ]

    # pipeline feature definition
    df["Feature"] = df[prot].astype(str) + "-" + df[prec].astype(str)

    # normalize MS2 signal
    df["MS2_norm"] = df[frag] / df[rep].map(psnf)

    reduced = df[[rep, "Feature", "MS2_norm"]].copy()
    reduced.columns = ["CodeName", "Feature", "MS2"]

    reduced_rows.append(reduced)


print("[STEP 2] Building reduced feature table")

df_reduced = pd.concat(reduced_rows, ignore_index=True)

df_reduced = (
    df_reduced
    .groupby(["CodeName", "Feature"], as_index=False)
    .agg({"MS2": "max"})
)

print("Unique features before restriction:", df_reduced["Feature"].nunique())


print("[STEP 3] Building DIA feature matrix")

matrix = df_reduced.pivot(
    index="CodeName",
    columns="Feature",
    values="MS2"
).fillna(0)

matrix = matrix.reset_index()

matrix["Type"] = matrix["CodeName"].apply(extract_type_from_codename)
matrix = matrix[~matrix["Type"].isin(EXCLUDE_TYPES)]


# ============================================================
# PIPELINE FEATURE RESTRICTION (Step 20)
# ============================================================

if RESTRICT_TO_PROTEIN_LIST and os.path.exists(PROTEIN_IDS_FILE):

    print("[PIPELINE] Restricting features using protein_ids.txt")

    with open(PROTEIN_IDS_FILE) as f:
        keep_ids = {line.strip() for line in f if line.strip()}

    feature_cols = [
        c for c in matrix.columns
        if c not in {"CodeName", "Type"}
    ]

    keep_cols = []

    for feat in feature_cols:
        acc = extract_accession(feat)
        if acc in keep_ids:
            keep_cols.append(feat)

    print("Features before restriction:", len(feature_cols))
    print("Features after restriction:", len(keep_cols))

    matrix = matrix[["CodeName", "Type"] + keep_cols]

matrix_out = matrix.rename(columns={"CodeName": "Replicate"})
matrix_file = os.path.join(OUT_DIR, "dia_feature_matrix.csv")
matrix_out.to_csv(matrix_file, index=False)
print("Matrix written:", matrix_file)

print("[STEP 4] AUROC feature audit")

def wilson_ci(p, n, z=1.96):
    """Wilson score confidence interval for a proportion."""
    if n == 0 or np.isnan(p):
        return np.nan, np.nan
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)

type_map = load_type_map(PROTEIN_TYPE_FILE)

features = [c for c in matrix.columns if c not in {"CodeName", "Type"}]

for type_code, accs in type_map.items():

    y_true = (matrix["Type"] == type_code).astype(int).values

    if y_true.sum() == 0:
        continue

    rows = []

    for feat in features:

        vals = matrix[feat].values

        pos_vals = vals[y_true == 1]
        neg_vals = vals[y_true == 0]

        if np.nanmax(vals) == 0:
            auc = np.nan
            best_cut = np.nan
            sens = np.nan
            spec = np.nan
            spec_cut = np.nan

        else:

            try:

                if np.all(neg_vals == 0) and np.any(pos_vals > 0):
                    auc = 1.0
                    best_cut = np.min(pos_vals[pos_vals > 0])

                else:
                    auc = roc_auc_score(y_true, vals)
                    fpr, tpr, thr = roc_curve(y_true, vals)
                    j = tpr - fpr
                    best_cut = thr[np.argmax(j)]

            except Exception:
                auc = np.nan
                best_cut = np.nan

            if not np.isnan(best_cut):
                sens = np.mean(pos_vals >= best_cut)
                spec = np.mean(neg_vals < best_cut)
            else:
                sens = np.nan
                spec = np.nan

            spec_cut = np.max(neg_vals) if len(neg_vals) else np.nan

        n_pos = len(pos_vals)
        n_neg = len(neg_vals)
        sens_lo, sens_hi = wilson_ci(sens, n_pos)
        spec_lo, spec_hi = wilson_ci(spec, n_neg)

        rows.append({
            "Feature": feat,
            "AUROC": auc,
            "Sensitivity": sens,
            "Sensitivity_CI_lower": sens_lo,
            "Sensitivity_CI_upper": sens_hi,
            "Specificity": spec,
            "Specificity_CI_lower": spec_lo,
            "Specificity_CI_upper": spec_hi,
            "N_pos": n_pos,
            "N_neg": n_neg,
            "BestCutoff": best_cut,
            "SpecificityCutoff": spec_cut,
            "BelongsToType": extract_accession(feat) in accs
        })

    df_out = pd.DataFrame(rows).sort_values("AUROC", ascending=False)

    out_path = os.path.join(OUT_DIR, f"Audit_{type_code}.csv")
    df_out.to_csv(out_path, index=False)

    print("Wrote", out_path)


print("[DONE]")

# ============================================================
# STEP 24 - CREATE RF TRAINING DATASET
# ============================================================
print("[STEP 24] Creating RF training dataset")

# Copy matrix
rf_df = matrix.copy()

# Preserve replicate identifier
rf_df = rf_df.rename(columns={"CodeName": "Replicate"})

# ------------------------------------------------------------
# Filter rows by ML-supported amyloid types
# ------------------------------------------------------------
before_rows = len(rf_df)
rf_df = rf_df[rf_df["Type"].isin(ML_TYPES)].copy()
after_rows = len(rf_df)

print(
    f"[FILTER] Rows by Type {sorted(ML_TYPES)}: "
    f"{before_rows} -> {after_rows}"
)

# ------------------------------------------------------------
# Optional: restrict RF features to peptides from amyloid proteins
# ------------------------------------------------------------
amyloid_accs = set()

if RESTRICT_RF_TO_AMYLOID_PROTEINS:
    df_pt = pd.read_csv(PROTEIN_TYPE_FILE, sep="\t", dtype=str).fillna("")
    for _, row in df_pt.iterrows():
        for val in row.iloc[1:]:
            if val.strip():
                amyloid_accs.add(val.strip().upper())

# Determine which feature columns to keep
keep_cols = ["Replicate", "Type"]

for col in rf_df.columns:
    if col in {"Replicate", "Type"}:
        continue

    acc = extract_accession(col)

    if not RESTRICT_RF_TO_AMYLOID_PROTEINS or (
        acc and acc.upper() in amyloid_accs
    ):
        keep_cols.append(col)

rf_df = rf_df[keep_cols]

print(
    f"[FILTER] Feature columns retained: {len(keep_cols)-1}"
)

# ------------------------------------------------------------
# Ensure numeric feature values
# ------------------------------------------------------------
feature_cols = [c for c in rf_df.columns if c not in {"Replicate", "Type"}]

rf_df[feature_cols] = (
    rf_df[feature_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0.0)
)

# ------------------------------------------------------------
# Write RF training dataset
# ------------------------------------------------------------
rf_out = os.path.join(OUT_DIR, "rf_training_data.csv")
rf_df.to_csv(rf_out, index=False)

print(f"[OK] RF training dataset written: {rf_out}")

# ============================================================
# STEP 5 - PACKAGE PEPTIDE THRESHOLD TABLES
# ============================================================

THRESH_DIR = os.path.join(OUT_DIR, "peptide_thresholds")
os.makedirs(THRESH_DIR, exist_ok=True)

# Move Audit files into peptide_thresholds directory
audit_files = [
    f for f in os.listdir(OUT_DIR)
    if f.startswith("Audit_") and f.endswith(".csv")
]

compiled_rows = []

for f in audit_files:

    src = os.path.join(OUT_DIR, f)
    dst = os.path.join(THRESH_DIR, f)

    # move file
    os.replace(src, dst)

    # read and annotate
    df = pd.read_csv(dst)

    type_code = f.replace("Audit_", "").replace(".csv", "")
    df["Type"] = type_code

    compiled_rows.append(df)

print("[STEP 5] Compiling peptide threshold tables")

audit_compiled = pd.concat(compiled_rows, ignore_index=True)

# ------------------------------------------------------------
# Apply filters
# ------------------------------------------------------------

audit_compiled = audit_compiled[
    audit_compiled["BelongsToType"] == True
]

audit_compiled = audit_compiled[
    audit_compiled["AUROC"] >= 0.50
]

# ------------------------------------------------------------
# Sort rows
# ------------------------------------------------------------

audit_compiled = audit_compiled.sort_values(
    by=["Type", "Sensitivity", "Specificity"],
    ascending=[True, False, False]
)

# ------------------------------------------------------------
# Rename Type codes to amyloid display names and re-sort
# ------------------------------------------------------------

TYPE_DISPLAY_NAMES = {
    "SAA": "AA",
    "AA1": "AApoA1",
    "B2M": "Abeta2M",
    "CAL": "ACal",
    "FIB": "AFib",
    "ILA": "AIL1KA",
    "INS": "AIns",
    "KER": "AKer",
    "ALK": "AL kappa",
    "ALL": "AL lambda",
    "LT2": "ALect2",
    "SEM": "ASem",
    "THY": "ATTR",
    "HMU": "Heavy mu",
}

audit_compiled["Type"] = audit_compiled["Type"].map(TYPE_DISPLAY_NAMES).fillna(audit_compiled["Type"])

audit_compiled["_type_sort"] = audit_compiled["Type"].str.lower()
audit_compiled = audit_compiled.sort_values(
    by=["_type_sort", "Sensitivity", "Specificity"],
    ascending=[True, False, False]
)
audit_compiled = audit_compiled.drop(columns=["_type_sort"])

# ------------------------------------------------------------
# Format CI columns for publication tables
# ------------------------------------------------------------

def _fmt_ci(row, val_col, lo_col, hi_col):
    return f"{row[val_col]:.3f} ({row[lo_col]:.3f}-{row[hi_col]:.3f})"

def format_for_table(df):
    df = df.copy()
    df["Sensitivity (95% CI)"] = df.apply(
        _fmt_ci, axis=1,
        val_col="Sensitivity", lo_col="Sensitivity_CI_lower", hi_col="Sensitivity_CI_upper"
    )
    df["Specificity (95% CI)"] = df.apply(
        _fmt_ci, axis=1,
        val_col="Specificity", lo_col="Specificity_CI_lower", hi_col="Specificity_CI_upper"
    )
    drop_cols = [
        "Sensitivity", "Sensitivity_CI_lower", "Sensitivity_CI_upper",
        "Specificity", "Specificity_CI_lower", "Specificity_CI_upper",
        "BelongsToType"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    col_order = [
        "Type", "Feature", "AUROC",
        "Sensitivity (95% CI)", "Specificity (95% CI)",
        "N_pos", "N_neg", "BestCutoff", "SpecificityCutoff"
    ]
    return df[[c for c in col_order if c in df.columns]]

compiled_path = os.path.join(THRESH_DIR, "audit_compiled.csv")

format_for_table(audit_compiled).to_csv(compiled_path, index=False)

print("Compiled audit table written:", compiled_path)

# ------------------------------------------------------------
# Create top-3 peptides per type
# ------------------------------------------------------------

audit_top3 = (
    audit_compiled
    .groupby("Type", group_keys=False)
    .head(3)
)

top3_path = os.path.join(THRESH_DIR, "audit_compiled_top3.csv")

format_for_table(audit_top3).to_csv(top3_path, index=False)

print("Top-3 peptide table written:", top3_path)
