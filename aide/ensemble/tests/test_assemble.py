"""Assemble the per-row training matrix from a synthetic cache: geometry joined by item,
label groups concatenated across OOF folds, y/subjects aligned, shard-order asserted."""
import numpy as np

from aide.ensemble.assemble import assemble_training_matrix
from aide.features.cache import FeatureCache
from aide.features.store import FoldFeatureStore
from aide.harness.funnel import FeatureBlock
from aide.hygiene.manifest import build_manifest
from aide.hygiene.splits import outer_folds


def _rid(s, i):
    return np.array([f"{a}|{b}" for a, b in zip(s, i)])


def test_assemble_joins_geometry_and_concats_label_folds(tmp_path):
    items = ["i0", "i1", "i2", "i3"]
    # full measurement rows (every subject × item); arbitrary order
    rows = [("s0", "i0", 1.0, "b0"), ("s1", "i0", 0.0, "b0"), ("s0", "i1", 1.0, "b1"),
            ("s1", "i2", 0.0, "b0"), ("s0", "i3", 1.0, "b1"), ("s1", "i3", 1.0, "b1"),
            ("s0", "i2", 0.0, "b0"), ("s1", "i1", 1.0, "b1")]
    subj = np.array([r[0] for r in rows]); item = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows]); bench = np.array([r[3] for r in rows])
    man = build_manifest(items, n_folds=2, seed=0)

    store = FoldFeatureStore(FeatureCache(tmp_path, code_version="t"), embedding_family="m",
                             seed=0, n_folds=2)
    # geometry: one row per item, value = item index (so we can verify the per-row join)
    gX = np.array([[float(j), float(j) * 10] for j in range(4)], dtype=np.float32)
    store.write_group("nn_geometry", "all",
                      FeatureBlock(X=gX, columns=["geo__a", "geo__b"], row_ids=np.array(items)),
                      inputs_hash="h")
    # label group per fold, in the SAME row order the assembler reconstructs
    for f in outer_folds(man):
        oof = set(f.oof_item_keys)
        mask = np.array([it in oof for it in item])
        rids = _rid(subj[mask], item[mask])
        lab = np.arange(mask.sum(), dtype=np.float32).reshape(-1, 1) + 100 * f.index
        store.write_group("nn_label_derivatives", f.index,
                          FeatureBlock(X=lab, columns=["nn__v"], row_ids=rids), inputs_hash="h")

    ds = assemble_training_matrix(store, man, item_keys=item, subject_keys=subj, labels=y,
                                  benchmarks=bench, geometry_groups=["nn_geometry"],
                                  label_groups=["nn_label_derivatives"])
    assert ds.X.shape == (8, 3)                       # geo__a, geo__b, nn__v
    assert ds.feature_columns == ["geo__a", "geo__b", "nn__v"]
    assert len(ds.y) == 8 and set(ds.subjects) == {"s0", "s1"}
    # every row's geometry equals ITS item's geometry (the join is correct)
    item_to_geo = {it: gX[j] for j, it in enumerate(items)}
    for r in range(len(ds.y)):
        assert np.allclose(ds.X[r, :2], item_to_geo[ds.item_keys[r]])
    # y stays aligned with the (subject,item) identity through the fold concatenation
    truth = {(s, i): yy for s, i, yy, _ in rows}
    for r in range(len(ds.y)):
        assert ds.y[r] == truth[(ds.subjects[r], ds.item_keys[r])]


def test_assemble_detects_shard_row_misalignment(tmp_path):
    items = ["i0", "i1"]
    subj = np.array(["s0", "s0"]); item = np.array(["i0", "i1"])
    y = np.array([1.0, 0.0]); bench = np.array(["b", "b"])
    man = build_manifest(items, n_folds=2, seed=0)
    store = FoldFeatureStore(FeatureCache(tmp_path, code_version="t"), embedding_family="m",
                             seed=0, n_folds=2)
    store.write_group("nn_geometry", "all",
                      FeatureBlock(X=np.zeros((2, 1), np.float32), columns=["geo__a"],
                                   row_ids=np.array(items)), inputs_hash="h")
    # write a label shard with WRONG row_ids for its fold
    for f in outer_folds(man):
        oof = set(f.oof_item_keys)
        mask = np.array([it in oof for it in item])
        if mask.any():
            store.write_group("nn_label_derivatives", f.index,
                              FeatureBlock(X=np.zeros((mask.sum(), 1), np.float32),
                                           columns=["nn__v"],
                                           row_ids=np.array(["WRONG|id"] * mask.sum())),
                              inputs_hash="h")
    import pytest
    with pytest.raises(ValueError):
        assemble_training_matrix(store, man, item_keys=item, subject_keys=subj, labels=y,
                                 benchmarks=bench, geometry_groups=["nn_geometry"],
                                 label_groups=["nn_label_derivatives"])
