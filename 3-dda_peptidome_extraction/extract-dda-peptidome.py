#!/usr/bin/env python3
"""
extract-peptidome.py

Identify dominant proteins from NSAF spectral count data and
extract corresponding peptide evidence from Percolator peptide
outputs.

Workflow
--------
1. Rank proteins by maximum NSAF across replicates
2. Select top N proteins
3. Extract UniProt accessions
4. Scan Percolator peptide outputs
5. Retain peptides mapping to selected proteins
6. Aggregate peptide evidence

Author: <your name>
"""

import os
import re
import glob
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Number of top proteins to extract
TOP_PROTEIN_COUNT = 500

# Percolator peptide q‑value threshold
QVALUE_THRESHOLD = 0.01

# Input files
SPECTRAL_COUNTS_FILE = "spectral-counts.txt"

# Directory containing Percolator peptide outputs
PERCOLATOR_DIR = "percolator_peptides"

# Percolator peptide filename pattern
PEPTIDE_FILE_PATTERN = "*_peptides.txt"

# Output directory
OUTPUT_DIR = "extracted-peptidome"


# ============================================================
# HELPERS
# ============================================================

def extract_accession(header):
    """
    Extract UniProt accession from FASTA-style header.

    Example:
        sp|P12345|PROT_HUMAN -> P12345
    """
    if pd.isna(header):
        return ""
    m = re.search(r"\|([A-Z0-9]+)\|", str(header))
    return m.group(1) if m else ""


# ============================================================
# STEP 1 — LOAD SPECTRAL COUNTS
# ============================================================

print("\n[STEP 1] Loading spectral counts")

if not os.path.exists(SPECTRAL_COUNTS_FILE):
    raise FileNotFoundError(f"Missing file: {SPECTRAL_COUNTS_FILE}")

df = pd.read_csv(SPECTRAL_COUNTS_FILE, sep="\t")
df = df.copy()

protein_col = df.columns[0]
desc_col = df.columns[1]
sample_cols = df.columns[2:]

df[sample_cols] = df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0)


# ============================================================
# STEP 2 — RANK PROTEINS BY NSAF
# ============================================================

print("[STEP 2] Ranking proteins")

df_ranked = (
    df.assign(NSAF_MAX=df[sample_cols].max(axis=1))
      .sort_values("NSAF_MAX", ascending=False)
      .reset_index(drop=True)
      .copy()
)

df_ranked.insert(0, "Rank", range(1, len(df_ranked) + 1))


# ============================================================
# STEP 3 — SELECT TOP PROTEINS
# ============================================================

print(f"[STEP 3] Selecting top {TOP_PROTEIN_COUNT} proteins")

df_top = df_ranked.head(TOP_PROTEIN_COUNT).copy()

df_top["Protein_ID"] = df_top[protein_col].apply(extract_accession)

top_ids = [pid for pid in df_top["Protein_ID"] if pid]
top_ids_set = set(top_ids)

protein_rank = {pid: i for i, pid in enumerate(top_ids)}

print(f"[INFO] Unique protein IDs: {len(top_ids)}")


# ============================================================
# STEP 4 — WRITE PROTEIN OUTPUTS
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

protein_table = os.path.join(OUTPUT_DIR, "top-proteins.tsv")
df_top.to_csv(protein_table, sep="\t", index=False)

print(f"[OK] Wrote {protein_table}")

id_file = os.path.join(OUTPUT_DIR, "top-protein-ids.txt")

with open(id_file, "w") as f:
    for pid in top_ids:
        f.write(pid + "\n")

print(f"[OK] Wrote {id_file}")

# ============================================================
# STEP 5 — LOAD AND FILTER PERCOLATOR PEPTIDES (OPTIMIZED)
# ============================================================
print("\n[STEP 5] Reading Percolator peptides")

pep_files = sorted(
    glob.glob(os.path.join(PERCOLATOR_DIR, PEPTIDE_FILE_PATTERN))
)

if not pep_files:
    raise RuntimeError("No Percolator peptide files found")

rows = []

if not top_ids:
    raise RuntimeError("No valid protein accessions were extracted from spectral-counts.txt")

# Filter by protein IDs using exact accession matching
def match_top_protein(cell):
    accessions = re.findall(r"\|([A-Z0-9\-]+)\|", cell)
    base_ids = [acc.split("-")[0] for acc in accessions]
    for acc in base_ids:
        if acc in top_ids_set:
            return acc
    return None

pep_protein_col = None

for fpath in pep_files:

    sample = os.path.basename(fpath).replace("_peptides.txt", "")

    try:
        df_pep = pd.read_csv(fpath, sep="\t", dtype=str).fillna("")
    except Exception as e:
        print(f"[WARN] Could not read {fpath}: {e}")
        continue

    cols_lower = {c.lower(): c for c in df_pep.columns}

    qvalue_col = None

    for key, col in cols_lower.items():
        if pep_protein_col is None and key.startswith("protein"):
            pep_protein_col = col
        if "q-value" in key or "qvalue" in key:
            qvalue_col = col

    if pep_protein_col is None:
        raise RuntimeError(f"Protein column not found in {fpath}")

    if qvalue_col is None:
        raise RuntimeError(f"q-value column not found in {fpath}")

    # Convert q-values
    df_pep[qvalue_col] = pd.to_numeric(df_pep[qvalue_col], errors="coerce")
    df_pep = df_pep[df_pep[qvalue_col].notna()]
    df_pep = df_pep[df_pep[qvalue_col] <= QVALUE_THRESHOLD]

    df_pep["matched_protein"] = df_pep[pep_protein_col].apply(match_top_protein)
    df_pep = df_pep[df_pep["matched_protein"].notna()].copy()

    if df_pep.empty:
        continue

    df_pep["Sample"] = sample
    rows.append(df_pep)

if not rows:
    raise RuntimeError("No peptides matched selected proteins")

df_pep = pd.concat(rows, ignore_index=True)

# ------------------------------------------------------------
# Identify peptide sequence column
# ------------------------------------------------------------
seq_col = None
for c in df_pep.columns:
    if "sequence" in c.lower():
        seq_col = c
        break

if seq_col is None:
    raise RuntimeError("Could not identify peptide sequence column")

# ------------------------------------------------------------
# Compute peptide counts
# ------------------------------------------------------------
df_pep["count"] = df_pep.groupby(seq_col)[seq_col].transform("size")

# ------------------------------------------------------------
# Assign protein rank
# ------------------------------------------------------------
df_pep["protein_rank"] = df_pep["matched_protein"].map(protein_rank)

# ------------------------------------------------------------
# Sort like the original pipeline
# ------------------------------------------------------------
df_pep = df_pep.sort_values(
    by=["protein_rank", "count", seq_col],
    ascending=[True, False, True]
)

print(f"[INFO] Extracted {len(df_pep)} peptide entries")


# ============================================================
# STEP 6 — BUILD PEPTIDE TABLE
# ============================================================

print("[STEP 6] Building peptide table")

peptide_table = os.path.join(OUTPUT_DIR, "top-protein-peptides.tsv")

df_pep.to_csv(peptide_table, sep="\t", index=False)

print(f"[OK] Wrote {peptide_table}")


# ============================================================
# STEP 7 — UNIQUE PEPTIDES
# ============================================================

print("[STEP 7] Extracting unique peptide sequences")

df_unique = df_pep.drop_duplicates(subset=[seq_col])

unique_file = os.path.join(
    OUTPUT_DIR,
    "top-protein-peptides-unique.tsv"
)

df_unique.to_csv(unique_file, sep="\t", index=False)

print(f"[OK] Wrote {unique_file}")


print("\n[DONE] Peptidome extraction complete.")

# ============================================================
# STEP 8 — SKYLINE FORMAT PEPTIDE LIST
# ============================================================
print("[STEP 8] Writing Skyline peptide format")

# Identify flanking AA column
flank_col = None
for c in df_pep.columns:
    if "flanking" in c.lower():
        flank_col = c
        break

if flank_col is None:
    raise RuntimeError("Flanking AA column not found")

prot_col = pep_protein_col

def first_protein(cell):
    acc = re.findall(r"\|([A-Z0-9\-]+)\|", cell)
    if not acc:
        return "UNKNOWN"
    return acc[0].split("-")[0]

def peptide_string(seq, flank):
    if len(flank) >= 2:
        return f"{flank[0]}.{seq}.{flank[1]}"
    return f"-.{seq}.-"

counts = df_pep.groupby(seq_col).size().to_dict()

df_unique["header"] = df_unique.apply(
    lambda r: f">ap|{first_protein(r[prot_col])}-x|{peptide_string(r[seq_col], r[flank_col])}|{counts.get(r[seq_col],1)}",
    axis=1
)

skyline_file = os.path.join(OUTPUT_DIR, "amyloid-peptidome.txt")

df_unique[[seq_col, "header"]].to_csv(
    skyline_file,
    sep="\t",
    index=False,
    header=False
)

print(f"[OK] Wrote {skyline_file}")