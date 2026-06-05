"""Tests for the metadata join + fold-split helpers (Prep B). Pure-python/numpy parts;
the pandas-backed load + global driver run on Colab. A tiny iterrows stand-in avoids pandas."""
import numpy as np

from aide.features.metadata import (age_bin, extract_subject_name, row_benchmark_meta,
                                     row_subject_meta, split_block_by_fold)
from aide.harness.funnel import FeatureBlock


class _DF:
    """Minimal DataFrame stand-in exposing iterrows() over a list of dict rows."""
    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        for i, r in enumerate(self._rows):
            yield i, r


def test_extract_subject_name():
    assert extract_subject_name("Name: BioMistral-7B") == "BioMistral-7B"
    assert extract_subject_name("Name: Claude 3.5 Sonnet\nYou are...") == "Claude 3.5 Sonnet"
    assert extract_subject_name("name: 0-hero/Matter-7B") == "0-hero/Matter-7B"
    assert extract_subject_name("") == "UNK"


def test_age_bin():
    assert list(age_bin([0, 180, 400, np.nan], bin_days=180)) == [0, 1, 2, -1]


def test_row_subject_meta_exact_suffix_and_unk():
    mi = _DF([{"name": "0-hero/Matter-7B", "organization": "0-hero",
               "family": "Matter 7B", "macro-family": "Matter"}])
    meta, cov = row_subject_meta(["0-hero/Matter-7B", "Matter-7B", "Ghost"], mi)
    assert list(meta["organization"]) == ["0-hero", "0-hero", "UNK"]   # exact, suffix, miss
    assert list(meta["macro_family"]) == ["Matter", "Matter", "UNK"]
    assert abs(cov - 2 / 3) < 1e-9


def test_row_benchmark_meta():
    bi = _DF([{"benchmark": "afrimedqa", "topic": "Medicine", "age": 548, "has_conditions": 1}])
    meta = row_benchmark_meta(["afrimedqa", "unknown_bench"], bi)
    assert list(meta["topic"]) == ["Medicine", "UNK"]
    assert meta["age_bin"][0] == str(int(548 // 180))   # 3
    assert meta["age_bin"][1] == "-1"


def test_split_block_by_fold_partitions_rows():
    blk = FeatureBlock(X=np.arange(8.0).reshape(4, 2).astype(np.float32), columns=["a", "b"],
                       row_ids=np.array(["r0", "r1", "r2", "r3"]))
    parts = split_block_by_fold(blk, np.array([0, 1, 0, 2]))
    assert set(parts) == {0, 1, 2}
    assert list(parts[0].row_ids) == ["r0", "r2"]
    assert parts[0].X.tolist() == [[0.0, 1.0], [4.0, 5.0]]
    assert list(parts[1].row_ids) == ["r1"]
    # every row lands in exactly one fold shard (coverage)
    assert sum(p.X.shape[0] for p in parts.values()) == 4
