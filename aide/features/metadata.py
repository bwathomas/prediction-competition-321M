"""Metadata join + global OOF tabular driver (Plan 4 Prep B).

Wires the metadata-groupby tabular groups (``grp_subj__*``, ``grp_bench__*``) and
``mean_encoded_subject`` into the pipeline. These differ from the embedding-derived groups
in two ways handled here:
  * they need a JOIN to ``data/metadata/{model_info,benchmark_info}.csv`` (subject →
    organization/family/macro-family; benchmark → topic/age);
  * the OOF target encoding is GLOBAL (leave-own-fold-out over ALL rows), so it is computed
    once over the full table and then SPLIT by the row's OOF fold into per-fold shards.

The subject↔model join is fragile (a subject's ``Name:`` line may omit the ``org/`` prefix
the CSV ``name`` carries), so it falls back to a suffix match and reports coverage — verify
coverage on Colab before trusting the subject-metadata groups.
"""
from __future__ import annotations

import re

import numpy as np

_NAME_RE = re.compile(r"name:\s*(.+)", re.IGNORECASE)


def extract_subject_name(subject_content: str) -> str:
    """The model name from a subject's content (the first ``Name: ...`` line)."""
    first = str(subject_content).splitlines()[0] if str(subject_content).strip() else ""
    m = _NAME_RE.match(first.strip())
    return (m.group(1).strip() if m else first.strip()) or "UNK"


def age_bin(age, bin_days: int = 180):
    """Coarse age bucket (categorical key for grp_bench__age_bin). NaN/unknown → -1."""
    a = np.asarray(age, dtype=float)
    out = np.where(np.isfinite(a), np.floor(a / bin_days), -1.0)
    return out.astype(int)


def build_name_lookup(model_info):
    """name → row dict, plus a suffix (after '/') → row fallback for partial-name subjects."""
    exact, suffix = {}, {}
    for _, r in model_info.iterrows():
        nm = str(r["name"])
        exact[nm] = r
        suffix[nm.split("/")[-1]] = r
    return exact, suffix


def row_subject_meta(subject_names, model_info):
    """Per-row {organization, family, macro_family} (UNK if unmatched) + coverage fraction."""
    exact, suffix = build_name_lookup(model_info)
    org, fam, macro = [], [], []
    hits = 0
    for nm in subject_names:
        r = exact.get(str(nm)) or suffix.get(str(nm).split("/")[-1])
        if r is not None:
            hits += 1
            org.append(str(r["organization"])); fam.append(str(r["family"]))
            macro.append(str(r["macro-family"]))
        else:
            org.append("UNK"); fam.append("UNK"); macro.append("UNK")
    coverage = hits / max(len(list(subject_names)), 1)
    return {"organization": np.array(org), "family": np.array(fam),
            "macro_family": np.array(macro)}, coverage


def row_benchmark_meta(benchmarks, benchmark_info):
    """Per-row {topic, age_bin} keyed off benchmark_info (UNK/-1 if unmatched)."""
    by_b = {str(r["benchmark"]): r for _, r in benchmark_info.iterrows()}
    topic, ages = [], []
    for b in benchmarks:
        r = by_b.get(str(b))
        topic.append(str(r["topic"]) if r is not None else "UNK")
        ages.append(float(r["age"]) if r is not None and np.isfinite(float(r["age"])) else np.nan)
    return {"topic": np.array(topic), "age_bin": age_bin(ages).astype(str)}


def split_block_by_fold(block, fold_ids):
    """Partition a FeatureBlock's rows by fold id → {fold: FeatureBlock}. The global OOF
    encoding is one block over all rows; each (group, fold) shard is the rows OOF in fold."""
    from aide.harness.funnel import FeatureBlock
    fold_ids = np.asarray(fold_ids)
    out = {}
    for f in np.unique(fold_ids):
        m = fold_ids == f
        out[int(f)] = FeatureBlock(X=block.X[m], columns=list(block.columns),
                                   row_ids=np.asarray(block.row_ids)[m])
    return out


# ---- global tabular driver (Colab) --------------------------------------------------
def load_metadata(repo_root="."):
    import pandas as pd
    from pathlib import Path
    base = Path(repo_root) / "data" / "metadata"
    return (pd.read_csv(base / "model_info.csv"), pd.read_csv(base / "benchmark_info.csv"))


def derive_tabular_global(*, store, labels_df, manifest, model_info, benchmark_info,
                          family, code_version, progress=None, smoothings=(2.0, 20.0)):
    """Compute the OOF metadata-groupby + subject-encoding groups over ALL rows, then write
    per-fold shards. ``labels_df`` needs subject_key, item_key, label, subject_content,
    benchmark. Returns subject-join coverage (validate it before trusting grp_subj__*)."""
    from aide.features.derive_tabular import derive_tabular
    from aide.features.driver import content_inputs_hash, _row_ids

    item_keys = labels_df["item_key"].astype(str).to_numpy()
    subj_keys = labels_df["subject_key"].astype(str).to_numpy()
    y = labels_df["label"].astype(float).to_numpy()
    fold_ids = np.array([manifest.fold_of(k) for k in item_keys])
    row_ids = _row_ids(subj_keys, item_keys)

    names = [extract_subject_name(c) for c in labels_df["subject_content"]]
    subject_meta, coverage = row_subject_meta(names, model_info)
    benchmark_meta = row_benchmark_meta(labels_df["benchmark"].astype(str).to_numpy(),
                                        benchmark_info)
    if progress:
        progress(f"tabular: subject-join coverage {coverage:.3f}", coverage=coverage)

    blocks = derive_tabular(row_ids=row_ids, fold_ids=fold_ids, y=y, subject_keys=subj_keys,
                            subject_meta=subject_meta, benchmark_meta=benchmark_meta,
                            parents=None, smoothings=smoothings)
    # interactions_subject needs derived parents (subject_mean/cluster_difficulty) → skip here
    for g in ["groupby_subject_metadata", "groupby_benchmark_metadata", "mean_encoded_subject"]:
        for fold, blk in split_block_by_fold(blocks[g], fold_ids).items():
            store.write_group(g, fold, blk,
                              inputs_hash=content_inputs_hash(family, g, fold, code_version))
    return {"subject_join_coverage": coverage}
