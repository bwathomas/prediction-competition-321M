"""Export the qwen features as an item-cold-start task for a (subprocess-isolated) AIDE run.

AIDE (aideml) is an autonomous LLM agent that searches a tree of *code* solutions. It shares
the ``aide`` import name with THIS project, so it must run as a separate subprocess on data
FILES — it never imports our package. Hygiene is enforced here, not trusted to the agent:

  * ``item_fold_split`` partitions rows so train and the held-out fold are ITEM-DISJOINT;
  * ``export_for_aide`` writes a train table (features + item_key + label) for AIDE to do
    its own GroupKFold-by-item validation, PLUS a secret held-out table (features + label
    kept aside) so we can independently re-score AIDE's winner on truly unseen items.

If AIDE ignores the grouped-CV instruction and memorizes items, the secret-holdout score
exposes it — the agent cannot fake an honest cold-start number it never saw.
"""
from __future__ import annotations

import numpy as np


def item_fold_split(item_keys, manifest, holdout_fold=0):
    """(train_mask, holdout_mask) that are ITEM-disjoint: holdout = rows whose item is in
    ``holdout_fold``, train = the rest. Asserts no item straddles the split."""
    fold_of = manifest.assignment
    row_fold = np.array([fold_of[str(k)] for k in item_keys])
    holdout = row_fold == holdout_fold
    train = ~holdout
    tr_items = set(np.asarray(item_keys)[train].astype(str))
    ho_items = set(np.asarray(item_keys)[holdout].astype(str))
    if tr_items & ho_items:
        raise AssertionError("item leakage: an item appears in both train and holdout")
    return train, holdout


def export_for_aide(ds, manifest, *, out_dir, secret_dir=None, holdout_fold=0):
    """Write ``train.parquet`` (features + item_key + label) and ``holdout_features.parquet``
    (features + item_key, NO label) to ``out_dir`` — the directory AIDE sees and copies into
    its workspace. The secret ``holdout_labels.parquet`` goes to ``secret_dir`` (a SIBLING by
    default, NOT under out_dir) so the agent can never read the answers. Colab (needs pyarrow)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path
    secret_dir = secret_dir or f"{out_dir}_secret"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(secret_dir).mkdir(parents=True, exist_ok=True)
    train, holdout = item_fold_split(ds.item_keys, manifest, holdout_fold)
    cols = list(ds.feature_columns)

    def _table(mask, with_label):
        d = {c: pa.array(ds.X[mask, j].astype(np.float32)) for j, c in enumerate(cols)}
        d["item_key"] = pa.array(np.asarray(ds.item_keys)[mask].astype(str))
        if with_label:
            d["label"] = pa.array(ds.y[mask].astype(np.float32))
        return pa.table(d)

    paths = {}
    paths["train"] = f"{out_dir}/train.parquet"
    pq.write_table(_table(train, True), paths["train"])
    paths["holdout_features"] = f"{out_dir}/holdout_features.parquet"
    pq.write_table(_table(holdout, False), paths["holdout_features"])
    paths["holdout_labels"] = f"{secret_dir}/holdout_labels.parquet"
    pq.write_table(pa.table({"item_key": pa.array(np.asarray(ds.item_keys)[holdout].astype(str)),
                             "label": pa.array(ds.y[holdout].astype(np.float32))}),
                   paths["holdout_labels"])
    paths["n_train"] = int(train.sum())
    paths["n_holdout"] = int(holdout.sum())
    return paths


AIDE_GOAL = (
    "Predict the probability that an AI subject passes a benchmark item (binary `label`), "
    "for ITEMS NOT SEEN IN TRAINING (item cold-start). The provided features are precomputed, "
    "leakage-safe, out-of-fold embedding/neighbour/cluster statistics; `item_key` identifies "
    "the benchmark item and MUST NOT be used as a feature. CRITICAL: validate with "
    "GroupKFold on `item_key` (sklearn GroupKFold, groups=item_key) so no item appears in both "
    "train and validation — a random split overstates accuracy and is invalid here."
)
AIDE_EVAL = "mean binary log loss (cross-entropy), lower is better, under GroupKFold(item_key)"
