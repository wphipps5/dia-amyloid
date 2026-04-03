#!/usr/bin/env python3
"""
nsaf-composite-index-calc.py

Calibration, computation, aggregation, and visualization of continuous
amyloid typing indices derived from protein-level normalized spectral
abundance factors (NSAF).

This script computes replicate-level composite indices using bounded
Hill-type transformations, aggregates indices to the CASE level, and
produces CASE-level Type-versus-Other visualizations and performance
metrics. The resulting indices are continuous quantitative scores
intended to support downstream visualization, thresholding, and
conservative amyloid typing decisions.

Required input files (working directory):
  - spectral-counts.txt
  - protein-type.txt
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import differential_evolution

# ============================================================
# CONFIGURATION
# ============================================================

# Input files (working directory)
NSAF_FILE = "spectral-counts.txt"
PROTEIN_TYPE_FILE = "protein-type.txt"

# Output root directory
OUT_ROOT = "output"

# Internal staging dirs (unchanged logic)
DIR_REPLICATE = os.path.join(OUT_ROOT, "_staging_replicate")
DIR_INDEX_PLOTS = os.path.join(OUT_ROOT, "_staging_index_tables")
DIR_CASE = os.path.join(OUT_ROOT, "_staging_case")
DIR_CASE_PLOTS = os.path.join(DIR_CASE, "_staging_case_plots")

for d in (OUT_ROOT, DIR_REPLICATE, DIR_INDEX_PLOTS, DIR_CASE, DIR_CASE_PLOTS):
    os.makedirs(d, exist_ok=True)

# Excluded amyloid categories
EXCLUDE_TYPES = {"INA", "UNK", "NON"}

# Hill parameter bounds
BOUNDS = [
    (0.5, 4.0),   # p
    (0.5, 4.0),   # q
    (1e-6, 1e-2)  # Kp
]

# ============================================================
# BLOCK 1 — REPLICATE-LEVEL INDEX CALIBRATION
# ============================================================

# ----------------------------
# HELPERS (unchanged)
# ----------------------------

def extract_protein_id(s):
    if not isinstance(s, str):
        return ""
    parts = s.split("|")
    return parts[1] if len(parts) >= 3 else ""


def extract_type_from_codename(cn):
    if not isinstance(cn, str):
        return ""
    if len(cn) < 7:
        return ""
    return cn[4:7]


def hill_abs(x, p, Kp):
    return (x ** p) / (x ** p + Kp)


def hill_rel(r, q):
    return (r ** q) / (r ** q + 1.0)


def compute_index(x, competitor, p, q, Kp):
    if x <= 0:
        return 0.0
    abs_term = hill_abs(x, p, Kp)
    if competitor <= 0:
        rel_term = 1.0
    else:
        rel_term = hill_rel(x / competitor, q)
    return abs_term * rel_term


def optimize_params(x, r_comp, labels):
    def objective(params):
        p, q, Kp = params
        scores = []
        for xi, ci in zip(x, r_comp):
            scores.append(compute_index(xi, ci, p, q, Kp))
        try:
            return -roc_auc_score(labels, scores)
        except ValueError:
            return 1.0

    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        strategy="best1bin",
        maxiter=200,
        polish=True,
        disp=False
    )
    return result.x


# ----------------------------
# LOAD DATA
# ----------------------------

print("[LOAD] NSAF table")
df = pd.read_csv(NSAF_FILE, sep="\t", dtype=str)
df["Protein_ID"] = df.iloc[:, 0].apply(extract_protein_id)

rep_cols = df.columns[2:]
valid_cols = []
for cn in rep_cols:
    t = extract_type_from_codename(cn)
    if t not in EXCLUDE_TYPES:
        valid_cols.append(cn)

long = df.melt(
    id_vars=["Protein_ID", "Description"],
    value_vars=valid_cols,
    var_name="CodeName",
    value_name="NSAF"
)
long["NSAF"] = pd.to_numeric(long["NSAF"], errors="coerce").fillna(0.0)


# ----------------------------
# LOAD PROTEIN GROUPS
# ----------------------------

ptype = pd.read_csv(PROTEIN_TYPE_FILE, sep="\t", dtype=str).fillna("")
ptype = ptype.set_index("Type")

ALL_PROT = set(ptype.loc["ALL"].values)
ALK_PROT = set(ptype.loc["ALK"].values)

FIB_A = "P02671"
FIB_B = "P02675"
FIB_C = "P02679"

KER_5 = "P13647"
KER_14 = "P02533"


# ----------------------------
# COMPUTE PRIMITIVES
# ----------------------------

print("[PRIMITIVES] Computing replicate-level NSAF features")
rows = []

for cn, grp in long.groupby("CodeName"):
    t = extract_type_from_codename(cn)

    lam = grp[grp["Protein_ID"].isin(ALL_PROT)]["NSAF"].max()
    kap = grp[grp["Protein_ID"].isin(ALK_PROT)]["NSAF"].max()

    fibA = grp.loc[grp["Protein_ID"] == FIB_A, "NSAF"].sum()
    fibB = grp.loc[grp["Protein_ID"] == FIB_B, "NSAF"].sum()
    fibC = grp.loc[grp["Protein_ID"] == FIB_C, "NSAF"].sum()
    fib_sum = fibA + fibB + fibC

    ker_bg = grp[
        grp["Description"].str.contains("keratin", case=False, na=False)
    ]["NSAF"].sum()

    ker5 = grp.loc[grp["Protein_ID"] == KER_5, "NSAF"].sum()
    ker14 = grp.loc[grp["Protein_ID"] == KER_14, "NSAF"].sum()
    ker_max = max(ker5, ker14)

    rows.append({
        "CodeName": cn,
        "Type": t,
        "lambda_max": lam,
        "kappa_max": kap,
        "fibA": fibA,
        "fib_sum": fib_sum,
        "ker_max": ker_max,
        "ker_sum": ker_bg
    })

prim = pd.DataFrame(rows)

before = len(prim)
prim = prim[~prim["Type"].isin(EXCLUDE_TYPES)].copy()
after = len(prim)
print(f"[FILTER] Excluded {before - after} INA/UNK/NON replicates")

prim.to_csv(
    os.path.join(DIR_REPLICATE, "replicate-primitives.tsv"),
    sep="\t",
    index=False
)


# ----------------------------
# TRAIN PARAMETERS
# ----------------------------

print("[TRAIN] Optimizing Hill parameters")
params = {}

def train_index(name, x_col, c_col, true_type):
    x = prim[x_col].values
    c = prim[c_col].values
    y = (prim["Type"] == true_type).astype(int).values
    if len(set(y)) < 2:
        raise RuntimeError(
            f"Training failed for {name}: only one class present"
        )
    p, q, Kp = optimize_params(x, c, y)
    params[name] = {"p": float(p), "q": float(q), "Kp": float(Kp)}

train_index("lambda", "lambda_max", "kappa_max", "ALL")
train_index("kappa", "kappa_max", "lambda_max", "ALK")
train_index("fib", "fibA", "fib_sum", "FIB")
train_index("ker", "ker_max", "ker_sum", "KER")

with open(os.path.join(DIR_REPLICATE, "trained-index-params.json"), "w") as f:
    json.dump(params, f, indent=2)


# ----------------------------
# COMPUTE INDICES
# ----------------------------

print("[INDEX] Computing indices")

def apply_index(row, key, x_col, c_col):
    p = params[key]["p"]
    q = params[key]["q"]
    Kp = params[key]["Kp"]
    return compute_index(row[x_col], row[c_col], p, q, Kp)

prim["lambda_index"] = prim.apply(
    lambda r: apply_index(r, "lambda", "lambda_max", "kappa_max"), axis=1
)
prim["kappa_index"] = prim.apply(
    lambda r: apply_index(r, "kappa", "kappa_max", "lambda_max"), axis=1
)
prim["fib_A_index"] = prim.apply(
    lambda r: apply_index(r, "fib", "fibA", "fib_sum"), axis=1
)
prim["K514_index"] = prim.apply(
    lambda r: apply_index(r, "ker", "ker_max", "ker_sum"), axis=1
)

prim["LK_diff"] = prim["lambda_index"] - prim["kappa_index"]


# ----------------------------
# DATASET-ANCHORED NORMALIZATION
# ----------------------------

norm_factors = {}

index_cols = {
    "lambda_index": "lambda",
    "kappa_index": "kappa",
    "fib_A_index": "fib",
    "K514_index": "ker",
}

for col, name in index_cols.items():
    max_val = prim[col].max()
    if max_val > 0:
        prim[col] = prim[col] / max_val
        norm_factors[name] = max_val
    else:
        norm_factors[name] = 0.0

prim["LK_diff"] = prim["lambda_index"] - prim["kappa_index"]

with open(
    os.path.join(DIR_REPLICATE, "index-normalization.json"), "w"
) as f:
    json.dump(norm_factors, f, indent=2)

print("[OK] Indices normalized to dataset max = 1.0")


# ----------------------------
# WRITE OUTPUTS
# ----------------------------

out_cols = [
    "CodeName", "Type",
    "lambda_index", "kappa_index", "LK_diff",
    "fib_A_index", "K514_index",
    "lambda_max", "kappa_max",
    "fibA", "fib_sum",
    "ker_max", "ker_sum"
]

prim[out_cols].to_csv(
    os.path.join(DIR_REPLICATE, "replicate-indices.tsv"),
    sep="\t",
    index=False
)

print("[DONE] Replicate-level index calibration complete")

# ============================================================
# BLOCK 2 — REPLICATE-LEVEL INDEX TABLE PREPARATION
# ============================================================

print("[BLOCK 2] Preparing replicate-level index tables")

# ----------------------------
# PATHS (adjusted only)
# ----------------------------

INPUT_FILE = os.path.join(
    DIR_REPLICATE,
    "replicate-indices.tsv"
)

OUT_DIR_13 = DIR_INDEX_PLOTS
os.makedirs(OUT_DIR_13, exist_ok=True)


# ----------------------------
# LOAD INPUT
# ----------------------------

df = pd.read_csv(INPUT_FILE, sep="\t", dtype=str).fillna("")


# ----------------------------
# FILTER TO RELEVANT AMYLOID TYPES
# ----------------------------

df["Type"] = df["CodeName"].str[4:7]
keep_types = {"ALL", "ALK", "KER", "FIB"}

before = len(df)
df = df[df["Type"].isin(keep_types)].copy()
after = len(df)

print(
    f"[FILTER] Retained {after} rows with types ALL/ALK/KER/FIB "
    f"(removed {before - after})"
)


# ----------------------------
# ENSURE NUMERIC COLUMNS
# ----------------------------

numeric_cols = [
    "fibA", "fib_sum", "fib_A_index",
    "kappa_max", "lambda_max", "kappa_index",
    "ker_max", "ker_sum", "K514_index",
    "lambda_index", "LK_diff"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)


# ----------------------------
# HELPER: WRITE TABLE
# ----------------------------

def write_table(df_sub, cols, sort_col, out_name):
    df_out = df_sub[cols].copy()
    df_out = df_out.sort_values(sort_col, ascending=False)
    out_path = os.path.join(OUT_DIR_13, out_name)
    df_out.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote {out_path} ({len(df_out)} rows)")


# ----------------------------
# FIBRINOGEN INDEX
# ----------------------------

write_table(
    df,
    cols=[
        "CodeName",
        "Type",
        "fibA",
        "fib_sum",
        "fib_A_index",
    ],
    sort_col="fib_A_index",
    out_name="fib-index.tsv"
)


# ----------------------------
# KAPPA INDEX
# ----------------------------

write_table(
    df,
    cols=[
        "CodeName",
        "Type",
        "kappa_max",
        "lambda_max",
        "kappa_index",
    ],
    sort_col="kappa_index",
    out_name="kappa-index.tsv"
)


# ----------------------------
# KERATIN INDEX
# ----------------------------

write_table(
    df,
    cols=[
        "CodeName",
        "Type",
        "ker_max",
        "ker_sum",
        "K514_index",
    ],
    sort_col="K514_index",
    out_name="ker-index.tsv"
)


# ----------------------------
# LAMBDA INDEX
# ----------------------------

write_table(
    df,
    cols=[
        "CodeName",
        "Type",
        "lambda_max",
        "kappa_max",
        "lambda_index",
    ],
    sort_col="lambda_index",
    out_name="lambda-index.tsv"
)


# ----------------------------
# LAMBDA − KAPPA DIFFERENCE
# ----------------------------

write_table(
    df,
    cols=[
        "CodeName",
        "Type",
        "lambda_index",
        "kappa_index",
        "LK_diff",
    ],
    sort_col="LK_diff",
    out_name="LK-diff.tsv"
)

print("[DONE] Replicate-level index table preparation complete")

# ============================================================
# BLOCK 2b — AUDIT-CODE INTEGRATION (CSV INPUT, CORRECTED)
# ============================================================

print("[BLOCK 2b] Applying Audit-Code annotations")

AUDIT_FILE = "audit-codes.csv"

# Map index table → target amyloid type
INDEX_TARGET_TYPE = {
    "lambda-index.tsv": "ALL",
    "kappa-index.tsv":  "ALK",
    "fib-index.tsv":    "FIB",
    "ker-index.tsv":    "KER",
    "LK-diff.tsv":      None,  # handled downstream
}

# Load user-provided audit codes if present
user_audit = {}

if os.path.exists(AUDIT_FILE):
    print(f"[LOAD] Using audit file: {AUDIT_FILE}")
    df_audit = pd.read_csv(AUDIT_FILE, sep=",", dtype=str).fillna("")
    if not {"CodeName", "Audit-Code"}.issubset(df_audit.columns):
        raise RuntimeError(
            "audit-codes.csv must contain columns: CodeName, Audit-Code"
        )
    user_audit = dict(zip(df_audit["CodeName"], df_audit["Audit-Code"]))
else:
    print("[INFO] No audit-codes.csv found — assuming all replicates are valid")

# Warn if audit file exists but does not cover all CodeNames
if user_audit:
    all_codenames = set(df["CodeName"])
    missing = all_codenames - set(user_audit.keys())
    if missing:
        print(
            f"[WARN] {len(missing)} CodeNames missing from audit-codes.csv; "
            "defaulting to '1' (technically valid)"
        )

# Apply audit codes to replicate-level index tables
for fname, target_type in INDEX_TARGET_TYPE.items():
    fpath = os.path.join(DIR_INDEX_PLOTS, fname)
    if not os.path.exists(fpath):
        continue

    df = pd.read_csv(fpath, sep="\t", dtype=str).fillna("")

    def assign_internal_audit(row):
        cn = row["CodeName"]
        user_code = user_audit.get(cn, "1")

        # Determine Type vs Other RELATIVE TO THIS INDEX
        if target_type is None:
            prefix = "T"  # LK-diff handled later
        else:
            prefix = "T" if row["Type"] == target_type else "O"

        return prefix + user_code

    df["Audit-Code"] = df.apply(assign_internal_audit, axis=1)

    # Derive Group-Code (legacy format)
    df["Group-Code"] = df["Type"] + "-" + df["Audit-Code"].str[1]

    df.to_csv(fpath, sep="\t", index=False)

print("[OK] Audit-Code and Group-Code applied (Type vs Other corrected)")


# ============================================================
# BLOCK 3 — CASE-LEVEL INDEX ROLLUP
# ============================================================

print("[BLOCK 3] Rolling up replicate-level indices to CASE level")

# ----------------------------
# PATHS (adjusted only)
# ----------------------------

BASE_DIR_14 = DIR_INDEX_PLOTS
OUT_DIR_14 = DIR_CASE
os.makedirs(OUT_DIR_14, exist_ok=True)

# Input files
single_index_files = {
    "fib":    "fib-index.tsv",
    "ker":    "ker-index.tsv",
    "lambda": "lambda-index.tsv",
    "kappa":  "kappa-index.tsv",
}

lk_file = "LK-diff.tsv"


# ----------------------------
# HELPERS (unchanged)
# ----------------------------

def extract_case(codename):
    """
    CASE = <Type>-<Block>
    Example: DDA-ALL0234-240227-A.1 → ALL-0234
    """
    m = re.match(r"DDA-([A-Z0-9]{3})(\d{4})", codename)
    if not m:
        raise RuntimeError(f"Cannot extract CASE from CodeName: {codename}")
    return f"{m.group(1)}-{m.group(2)}"


def resolve_audit_code(codes):
    """
    Resolve CASE-level Audit-Code from replicate-level Audit-Codes.
    Rules:
    - Type codes (T*) use: TF > TM > TL > TC > T1
    - Other codes (O*) use: OF > OM > OL > OC > O1
    """
    type_priority  = ["TF", "TM", "TL", "TC", "T1"]
    other_priority = ["OF", "OM", "OL", "OC", "O1"]

    for p in type_priority:
        if p in codes:
            return p

    for p in other_priority:
        if p in codes:
            return p

    return ""


def resolve_group_code(group_codes):
    """
    Legacy Group-Code priority: F > M > L > C > 1
    """
    priority = ["F", "M", "L", "C", "1"]
    types = {gc.split("-")[0] for gc in group_codes}

    if len(types) != 1:
        raise RuntimeError(f"Mixed Types in CASE group: {group_codes}")

    type_code = types.pop()
    code_letters = [gc.split("-")[1] for gc in group_codes]

    for p in priority:
        if p in code_letters:
            return f"{type_code}-{p}"

    raise RuntimeError(f"Unable to resolve Group-Code from {group_codes}")


# ----------------------------
# SINGLE-INDEX CASE ROLLUP
# ----------------------------

print("[CASE] Rolling up single-index tables")

for key, fname in single_index_files.items():
    print(f"[CASE] {fname}")

    df = pd.read_csv(
        os.path.join(BASE_DIR_14, fname),
        sep="\t",
        dtype=str
    ).fillna("")

    df["CASE"] = df["CodeName"].apply(extract_case)

    # Identify numeric index column
    idx_col = [c for c in df.columns if c.endswith("_index")][0]
    df[idx_col] = pd.to_numeric(df[idx_col], errors="coerce").fillna(0.0)

    rows = []

    for case, grp in df.groupby("CASE"):
        values = grp[idx_col].values
        mean_val = float(np.mean(values))
        rep_count = len(values)
        audit_codes = grp["Audit-Code"].tolist()

        audit_clean = resolve_audit_code(audit_codes)

        rows.append({
            "CASE": case,
            idx_col: mean_val,
            "Replicate_Count": rep_count,
            "Audit-Code": audit_clean
        })

    out_df = pd.DataFrame(rows)

    out_path = os.path.join(OUT_DIR_14, f"{key}-index-by-CASE.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)

    print(f"[OK] Wrote {out_path}")


# ----------------------------
# LK-DIFF CASE ROLLUP
# ----------------------------

print("[CASE] Rolling up LK-diff")

df = pd.read_csv(
    os.path.join(BASE_DIR_14, lk_file),
    sep="\t",
    dtype=str
).fillna("")

df["CASE"] = df["CodeName"].apply(extract_case)
df["LK_diff"] = pd.to_numeric(df["LK_diff"], errors="coerce")
df["LK_diff_adj"] = pd.to_numeric(df.get("LK_diff_adj", ""), errors="coerce")

rows = []

for case, grp in df.groupby("CASE"):
    group_codes = grp["Group-Code"].tolist()

    case_group = resolve_group_code(group_codes)

    keep = grp[~grp["Group-Code"].str.endswith("-F")]
    if keep.empty:
        continue

    mean_lk = float(np.mean(keep["LK_diff"]))
    mean_lk_adj = float(np.mean(keep["LK_diff_adj"])) if "LK_diff_adj" in keep else np.nan

    rows.append({
        "CASE": case,
        "LK_diff": mean_lk,
        "LK_diff_adj": mean_lk_adj,
        "Replicate_Count_Total": len(grp),
        "Replicate_Count_Used": len(keep),
        "Group-Code": case_group
    })

out_df = pd.DataFrame(rows)

out_path = os.path.join(OUT_DIR_14, "LK-diff-by-CASE.tsv")
out_df.to_csv(out_path, sep="\t", index=False)

print(f"[OK] Wrote {out_path}")
print("[DONE] CASE-level rollup complete")

# ============================================================
# BLOCK 4 — CASE-LEVEL INDEX PLOTS
# ============================================================

print("[BLOCK 4] Generating CASE-level plots")

# ----------------------------
# PATHS (adjusted only)
# ----------------------------

BASE_DIR_15 = DIR_CASE
OUT_DIR_15  = DIR_CASE_PLOTS
os.makedirs(OUT_DIR_15, exist_ok=True)


# ----------------------------
# STYLE MAPS (UNCHANGED)
# ----------------------------

single_style_map = {
    "T1": {"StyleName": "Type",          "marker": "o", "facecolor": "blue", "edgecolor": "gray",  "size": 220},
    "TC": {"StyleName": "Type",          "marker": "o", "facecolor": "black", "edgecolor": "gray",  "size": 220},
    "TL": {"StyleName": "Type-Limited",  "marker": "o", "facecolor": "white", "edgecolor": "black", "size": 120},
    "TF": {"StyleName": "Type-Fail",     "marker": "X", "facecolor": "white", "edgecolor": "black", "size": 160},
    "TM": {"StyleName": "Mixed",         "marker": "H", "facecolor": "gray",  "edgecolor": "white", "size": 250},
    "O1": {"StyleName": "Other",         "marker": "s", "facecolor": "black", "edgecolor": "gray",  "size": 140},
    "OM": {"StyleName": "Mixed",         "marker": "H", "facecolor": "gray",  "edgecolor": "white", "size": 250},
    "OL": {"StyleName": "Other-Limited", "marker": "s", "facecolor": "white", "edgecolor": "black", "size": 120},
    "OF": {"StyleName": "Other-Fail",    "marker": "X", "facecolor": "white", "edgecolor": "black", "size": 160},
    "OC": {"StyleName": "Carryover",     "marker": "P", "facecolor": "black", "edgecolor": "white", "size": 180},
}

group_style_map = {
    "AB4-1": {"StyleName":"Other","marker":"s","facecolor":"black","edgecolor":"gray","size":100},
    "FIB-1": {"StyleName":"Other","marker":"s","facecolor":"black","edgecolor":"gray","size":100},
    "HMU-1": {"StyleName":"Other","marker":"s","facecolor":"black","edgecolor":"gray","size":100},
    "KER-1": {"StyleName":"Other","marker":"s","facecolor":"black","edgecolor":"gray","size":100},
    "ALK-1": {"StyleName":"Type","marker":"^","facecolor":"black","edgecolor":"blue","size":180},
    "ALK-L": {"StyleName":"Type-Fail","marker":"X","facecolor":"white","edgecolor":"black","size":55},
    "ALK-F": {"StyleName":"Type-Fail","marker":"X","facecolor":"white","edgecolor":"black","size":55},
    "ALK-M": {"StyleName":"Mixed","marker":"H","facecolor":"gray","edgecolor":"white","size":150},
    "ALK-C": {"StyleName":"Carryover","marker":"P","facecolor":"black","edgecolor":"white","size":55},
    "ALL-1": {"StyleName":"Type","marker":"v","facecolor":"black","edgecolor":"red","size":180},
    "ALL-F": {"StyleName":"Type-Fail","marker":"X","facecolor":"white","edgecolor":"black","size":55},
    "ALL-L": {"StyleName":"Type-Fail","marker":"X","facecolor":"white","edgecolor":"black","size":55},
    "ALL-C": {"StyleName":"Carryover","marker":"P","facecolor":"black","edgecolor":"white","size":55},
    "AXX-1": {"StyleName":"Mixed","marker":"H","facecolor":"gray","edgecolor":"white","size":55},
    "AXX-F": {"StyleName":"Mixed","marker":"H","facecolor":"gray","edgecolor":"white","size":55},
    "AXX-M": {"StyleName":"Mixed","marker":"H","facecolor":"gray","edgecolor":"white","size":55},
    "MIX-M": {"StyleName":"Mixed","marker":"H","facecolor":"gray","edgecolor":"white","size":55},
}


# ----------------------------
# CONFIG FOR CASE-LEVEL PLOTS
# ----------------------------

plot_configs = {
    "lambda": {
        "file": "lambda-index-by-CASE.tsv",
        "value_col": "lambda_index",
        "type_code": "ALL",
        "ylabel": "Lambda Index (CASE)",
        "title": "Lambda Index (CASE)",
    },
    "kappa": {
        "file": "kappa-index-by-CASE.tsv",
        "value_col": "kappa_index",
        "type_code": "ALK",
        "ylabel": "Kappa Index (CASE)",
        "title": "Kappa Index (CASE)",
    },
    "fib": {
        "file": "fib-index-by-CASE.tsv",
        "value_col": "fib_A_index",
        "type_code": "FIB",
        "ylabel": "FibA Index (CASE)",
        "title": "Fibrinogen A Index (CASE)",
    },
    "ker": {
        "file": "ker-index-by-CASE.tsv",
        "value_col": "K514_index",
        "type_code": "KER",
        "ylabel": "Keratin 5/14 Index (CASE)",
        "title": "Keratin 5/14 Index (CASE)",
    },
}

rng = np.random.default_rng(12345)


# ----------------------------
# SINGLE-INDEX CASE PLOTS
# ----------------------------

print("[CASE-PLOT] Generating single-index CASE plots")

for key, cfg in plot_configs.items():
    print(f"[CASE-PLOT] {key}")

    df = pd.read_csv(
        os.path.join(BASE_DIR_15, cfg["file"]),
        sep="\t",
        dtype=str
    ).fillna("")

    df[cfg["value_col"]] = pd.to_numeric(
        df[cfg["value_col"]],
        errors="coerce"
    ).fillna(0.0)

    df["Category"] = np.where(
        df["CASE"].str.startswith(cfg["type_code"]),
        "Type",
        "Other"
    )

    for col in ["StyleName","marker","facecolor","edgecolor","size"]:
        df[col] = None
        df[col] = df[col].astype("object")

    for idx, row in df.iterrows():
        code = row["Audit-Code"]
        if code not in single_style_map:
            raise RuntimeError(
                f"No style for Audit-Code '{code}' in {cfg['file']}"
            )
        for k, v in single_style_map[code].items():
            df.at[idx, k] = v

    df["x_pos"] = df["Category"].map({"Type": 1, "Other": 2})
    df["x"] = df["x_pos"] + rng.uniform(-0.15, 0.15, len(df))

    fig, ax = plt.subplots(figsize=(7,6))

    for sty in df["StyleName"].unique():
        sub = df[df["StyleName"] == sty]
        if sub.empty:
            continue
        ax.scatter(
            sub["x"],
            sub[cfg["value_col"]],
            marker=sub["marker"].iloc[0],
            s=float(sub["size"].iloc[0]),
            facecolors=sub["facecolor"].iloc[0],
            edgecolors=sub["edgecolor"].iloc[0],
            alpha=0.85,
            label=sty
        )

    ax.set_xlim(0.5,2.5)
    ax.set_xticks([1,2])
    ax.set_xticklabels(["Type","Other"], fontweight="bold")
    ax.set_ylabel(cfg["ylabel"], fontweight="bold")
    ax.set_title(cfg["title"], fontweight="bold")

    if key in {"fib", "ker"}:
        o1_vals = df.loc[df["Audit-Code"] == "O1", cfg["value_col"]]
        if not o1_vals.empty:
            cutoff_o1 = o1_vals.max()
            ax.axhline(
                cutoff_o1,
                linestyle=":",
                linewidth=2,
                color="black",
                alpha=0.9,
                label=f"O1 max = {cutoff_o1:.3g}"
            )

    leg = ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize=8)
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR_15, f"{key}-CASE.png")
    plt.savefig(out_png, dpi=300)

    leg.remove()
    plt.tight_layout()
    plt.savefig(out_png.replace(".png","-noLegend.png"), dpi=300)
    plt.close()

    df.to_csv(
        os.path.join(OUT_DIR_15, f"{key}-CASE.tsv"),
        sep="\t",
        index=False
    )

    print(f"[OK] Wrote {out_png}")


# ----------------------------
# LK-DIFF CASE PLOT
# ----------------------------

print("[CASE-PLOT] Generating CASE-level LK-diff plot")

lk_case_path = os.path.join(BASE_DIR_15, "LK-diff-by-CASE.tsv")
df = pd.read_csv(lk_case_path, sep="\t", dtype=str).fillna("")

df["LK_diff"] = pd.to_numeric(df["LK_diff"], errors="coerce")

alk_clean = df[
    (df["CASE"].str.startswith("ALK-")) &
    (df["Group-Code"] == "ALK-1")
].copy()

if alk_clean.empty:
    raise RuntimeError(
        "[LK CUTOFF ERROR] No ALK-1 cases found to define cutoff"
    )

best_cutoff = alk_clean["LK_diff"].max()
print(f"[INFO] CASE-level LK-diff cutoff: {best_cutoff:.6f}")

df["LK_diff_adj"] = pd.to_numeric(df["LK_diff_adj"], errors="coerce")

for col in ["StyleName","marker","facecolor","edgecolor","size"]:
    df[col] = None
    df[col] = df[col].astype("object")

for idx, row in df.iterrows():
    gcode = row["Group-Code"]
    if gcode not in group_style_map:
        raise RuntimeError(f"No group style for '{gcode}'")
    for k,v in group_style_map[gcode].items():
        df.at[idx, k] = v

df["x"] = 1 + rng.uniform(-0.30, 0.30, len(df))

fig, ax = plt.subplots(figsize=(6,6))

for gcode in df["Group-Code"].unique():
    sub = df[df["Group-Code"] == gcode]
    ax.scatter(
        sub["x"],
        sub["LK_diff"],
        marker=sub["marker"].iloc[0],
        s=float(sub["size"].iloc[0]),
        facecolors=sub["facecolor"].iloc[0],
        edgecolors=sub["edgecolor"].iloc[0],
        alpha=0.85,
        label=gcode
    )

ax.axhline(0, linestyle="--", color="gray", alpha=0.4)
ax.axhline(
    best_cutoff,
    linestyle=":",
    linewidth=2,
    color="black",
    alpha=0.9,
    label=f"Cutoff = {best_cutoff:.3g}"
)

ax.set_xlim(0.6,1.4)
ax.set_xticks([1])
ax.set_xticklabels(["Lambda − Kappa (CASE)"], fontweight="bold")
ax.set_ylabel("Lambda − Kappa Index", fontweight="bold")
ax.set_title("Lambda–Kappa Index Differential (CASE)", fontweight="bold")

leg = ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize=8)
plt.tight_layout()

out_png = os.path.join(OUT_DIR_15, "LK-diff-CASE.png")
plt.savefig(out_png, dpi=300)

leg.remove()
plt.tight_layout()
plt.savefig(out_png.replace(".png","-noLegend.png"), dpi=300)
plt.close()

df.to_csv(
    os.path.join(OUT_DIR_15, "LK-diff-CASE.tsv"),
    sep="\t",
    index=False
)

print("[DONE] CASE-level plotting complete")

# ============================================================
# BLOCK 5 — FINAL OUTPUT ORGANIZATION (PUBLICATION STRUCTURE)
# ============================================================

print("[BLOCK 5] Organizing final outputs")

import shutil

# ------------------------------------------------------------
# Final publication directories
# ------------------------------------------------------------
REPL_OUT = os.path.join(OUT_ROOT, "index_calc_by_repl")
CASE_OUT = os.path.join(OUT_ROOT, "index_calc_by_case")
CASE_TABLES = os.path.join(CASE_OUT, "tables")
CASE_PLOTS = os.path.join(CASE_OUT, "plots")

for d in (REPL_OUT, CASE_OUT, CASE_TABLES, CASE_PLOTS):
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------
# Replicate-level outputs
# ------------------------------------------------------------

# Core replicate outputs
shutil.move(
    os.path.join(DIR_REPLICATE, "replicate-primitives.tsv"),
    os.path.join(REPL_OUT, "primitives.tsv")
)
shutil.move(
    os.path.join(DIR_REPLICATE, "replicate-indices.tsv"),
    os.path.join(REPL_OUT, "replicate-indices.tsv")
)
shutil.move(
    os.path.join(DIR_REPLICATE, "index-normalization.json"),
    os.path.join(REPL_OUT, "index-normalization.json")
)

# Per-index replicate tables
for name in ["lambda", "kappa", "fib", "ker", "LK-diff"]:
    src = os.path.join(DIR_INDEX_PLOTS, f"{name}-index.tsv")
    if os.path.exists(src):
        dst = os.path.join(REPL_OUT, f"{name}-by_repl.tsv")
        shutil.move(src, dst)

# ------------------------------------------------------------
# Case-level annotated tables (keep ONLY annotated)
# ------------------------------------------------------------

for name in ["lambda", "kappa", "fib", "ker", "LK-diff"]:
    src = os.path.join(DIR_CASE_PLOTS, f"{name}-CASE.tsv")
    if os.path.exists(src):
        dst = os.path.join(CASE_TABLES, f"{name}-by_case.tsv")
        shutil.move(src, dst)

# ------------------------------------------------------------
# Case-level plots
# ------------------------------------------------------------

for fname in os.listdir(DIR_CASE_PLOTS):
    if fname.lower().endswith(".png"):
        shutil.move(
            os.path.join(DIR_CASE_PLOTS, fname),
            os.path.join(CASE_PLOTS, fname.replace("-CASE", "-by_case"))
        )

print("[DONE] Final output structure complete")

# ------------------------------------------------------------
# Remove staging directories (recursive, publication-safe)
# ------------------------------------------------------------
print("[CLEANUP] Removing staging directories")

for d in [
    DIR_CASE,
    DIR_INDEX_PLOTS,
    DIR_REPLICATE,
]:
    if os.path.exists(d):
        try:
            shutil.rmtree(d)
            print(f"[CLEAN] Removed staging directory: {d}")
        except Exception as e:
            print(f"[WARN] Could not remove {d}: {e}")