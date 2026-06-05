# Cleanup + Hygiene Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the `clean/aide-stacked-ensemble` branch, autonomously remove self-evidently stale artifacts, then build the `aide/hygiene/` package — item-uniform OOF splits, a cross-agent-shared `SplitManifest`, recursive (nested) inner folds, a proxy-aware subject/benchmark dropout, and leakage tripwires — with full unit-test coverage.

**Architecture:** Pure-Python + numpy package under `aide/hygiene/`, no GPU/Colab/Drive dependency. Item→fold assignment is a deterministic hash of `(seed, item_key)` so all three agents reproduce byte-identical folds. Dropout masks a proxy node *and all its descendants* atomically, keyed by entity so every row of a dropped subject/benchmark is masked consistently. Leakage probes are assertions any later harness can call on every reported score.

**Tech Stack:** Python 3.11, numpy, pytest, stdlib (`hashlib`, `json`, `dataclasses`). This is the spec's §3 `aide/hygiene/` + §4 + §11-cleanup, decoupled from anything that needs an A100.

**Spec:** `docs/superpowers/specs/2026-06-04-aide-stacked-ensemble-design.md`

---

## File Structure

```
aide/
  __init__.py                 # empty package marker
  hygiene/
    __init__.py               # re-exports the public API
    manifest.py               # SplitManifest, build_manifest, item_fold, assert_identical
    splits.py                 # Fold, outer_folds, inner_folds, row_fold_ids
    proxy_tree.py             # PROXY_TREE data + descendants() + all_masked_columns()
    dropout.py                # apply_proxy_dropout (entity-keyed, atomic node+descendants)
    probes.py                 # assert_item_disjoint, assert_row_uniform_safe, assert_no_proxy_leak
    tests/
      __init__.py
      test_manifest.py
      test_splits.py
      test_proxy_tree.py
      test_dropout.py
      test_probes.py
```

**Reuse note:** logic is ported/adapted from `src/oof_folds.py` (item-grouped folds, leakage invariants) and `src/data.py` (`item_key`). We do not import `src/` — `aide/` is the new canon; `src/` stays untouched in Plan 1 and is triaged during the ensemble plan.

**Test runner:** from the repo root, `python -m pytest aide/hygiene/tests -q`. Assumes a venv with `numpy` + `pytest` (already in `requirements.txt`). If no venv: `python -m venv .venv && . .venv/bin/activate && pip install numpy pytest`.

---

## Task 0: Autonomous cleanup of self-evidently stale artifacts

**Files:** deletions only (git-tracked; recoverable from history on `main`).

**Delete list (self-evidently stale — build artifacts, dead probes, ship-phase one-offs):**
- `ensemble_bundles/` — 6 prebuilt submission zips, 141 MB, regenerable build artifacts.
- `a100_ablation_notebook (6).ipynb` — stray numbered scratch copy at repo root (1.1 MB).
- `notebooks/moe_poc.{ipynb,py}`, `notebooks/rich_mlp_moe_probe.{ipynb,py}`, `notebooks/proxy_probe.{ipynb,py}`, `notebooks/loss_diversity_probe.{ipynb,py}` — dead experimental probes (per git log).
- `bisection/` — batching-bisection ship-runtime debugging dirs (out of scope; superseded).
- `Prediction-Competition/` — near-empty stray dir (only `.gitattributes`).
- `scripts/pack_*.py`, `scripts/_sim_*.py`, `scripts/_smoke_*.py`, `scripts/_audit_bundle.py`, `scripts/_bench_apply_batch.py`, `scripts/_check_minimal_nb.py`, `scripts/_perbc_cal_sources.py`, `scripts/_py_to_ipynb.py`, `scripts/audit_notebook.py`, `scripts/diag_state_dict.py`, `scripts/drop_labeling.py`, `scripts/inspect_turbo_judge.py`, `scripts/refactor_m2_to_mlp.py` — one-off packaging/sim/smoke scripts for the OLD ship pipeline (this phase does not ship).

**Keep (NOT self-evidently stale — triaged in later plans):** `src/`, `validation_harness/`, `docs/`, `configs/`, `data/`, `requirements.txt`, `README.md`, `starting_kit/`, `Google_Collab_harness/`, `tests/`, `notebooks/{a100_ablation,qwen8b_four_member_stacked,qwen8b_minimalist,ensemble_builder}.*`, `scripts/{build_ipynb_from_py,build_ensemble_submission,smoke_test_submission}.py`.

- [ ] **Step 1: Delete the stale artifacts**

```bash
cd /home/akhaemenid/projects/prediction-competition-321M
git rm -r --quiet \
  ensemble_bundles \
  "a100_ablation_notebook (6).ipynb" \
  notebooks/moe_poc.ipynb notebooks/moe_poc.py \
  notebooks/rich_mlp_moe_probe.ipynb notebooks/rich_mlp_moe_probe.py \
  notebooks/proxy_probe.ipynb notebooks/proxy_probe.py \
  notebooks/loss_diversity_probe.ipynb notebooks/loss_diversity_probe.py \
  bisection \
  Prediction-Competition
git rm --quiet scripts/pack_*.py scripts/_sim_*.py scripts/_smoke_*.py \
  scripts/_audit_bundle.py scripts/_bench_apply_batch.py scripts/_check_minimal_nb.py \
  scripts/_perbc_cal_sources.py scripts/_py_to_ipynb.py scripts/audit_notebook.py \
  scripts/diag_state_dict.py scripts/drop_labeling.py scripts/inspect_turbo_judge.py \
  scripts/refactor_m2_to_mlp.py
```

- [ ] **Step 2: Verify no surviving KEEP file imports a deleted module**

Run:
```bash
cd /home/akhaemenid/projects/prediction-competition-321M
git grep -nE "from (bisection|scripts\.(pack|_sim|_smoke|audit_notebook))|import (bisection)" -- 'src/**' 'notebooks/a100_ablation*' 'notebooks/qwen8b*' 'notebooks/ensemble_builder*' 'validation_harness/**' 'tests/**' || echo "NO DANGLING REFS"
```
Expected: `NO DANGLING REFS`. If any reference prints, it points at a KEEP file that needs the deleted module — stop and reassess that file before continuing.

- [ ] **Step 3: Commit**

```bash
git commit -q -m "chore: remove self-evidently stale artifacts (bundles, dead probes, ship one-offs)"
```

---

## Task 1: `aide` package scaffold + `SplitManifest`

**Files:**
- Create: `aide/__init__.py`, `aide/hygiene/__init__.py`, `aide/hygiene/tests/__init__.py`
- Create: `aide/hygiene/manifest.py`
- Test: `aide/hygiene/tests/test_manifest.py`

- [ ] **Step 1: Write the failing test**

```python
# aide/hygiene/tests/test_manifest.py
from aide.hygiene.manifest import build_manifest, item_fold, assert_identical


def test_item_fold_is_deterministic_and_in_range():
    f1 = item_fold("benchA\nnone\nq1", n_folds=3, seed=0)
    f2 = item_fold("benchA\nnone\nq1", n_folds=3, seed=0)
    assert f1 == f2
    assert 0 <= f1 < 3


def test_build_manifest_assigns_every_unique_item_once():
    keys = ["a", "a", "b", "c", "c", "c"]
    m = build_manifest(keys, n_folds=3, seed=7)
    assert set(m.assignment) == {"a", "b", "c"}
    assert all(0 <= v < 3 for v in m.assignment.values())


def test_two_agents_same_seed_produce_identical_manifest():
    keys = ["a", "b", "c", "d", "e"]
    m_llama = build_manifest(keys, n_folds=3, seed=0)
    m_qwen = build_manifest(list(reversed(keys)), n_folds=3, seed=0)  # different order
    assert_identical(m_llama, m_qwen)  # must NOT raise — order-independent


def test_save_load_roundtrip(tmp_path):
    from aide.hygiene.manifest import SplitManifest
    m = build_manifest(["a", "b", "c"], n_folds=3, seed=1)
    p = tmp_path / "manifest.json"
    m.save(p)
    m2 = SplitManifest.load(p)
    assert_identical(m, m2)


def test_assert_identical_raises_on_mismatch():
    import pytest
    a = build_manifest(["x", "y"], n_folds=3, seed=0)
    b = build_manifest(["x", "y"], n_folds=3, seed=1)
    with pytest.raises(AssertionError):
        assert_identical(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest aide/hygiene/tests/test_manifest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aide'`.

- [ ] **Step 3: Write minimal implementation**

```python
# aide/__init__.py
```
```python
# aide/hygiene/tests/__init__.py
```
```python
# aide/hygiene/manifest.py
"""Canonical, cross-agent OOF split manifest.

Item->fold assignment is a deterministic hash of (seed, item_key) so all three
agents (llama/qwen/mistral) reproduce BYTE-IDENTICAL folds without sharing state.
``assert_identical`` is the cross-agent leakage guard.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def item_fold(item_key: str, n_folds: int, seed: int) -> int:
    """Deterministic fold id for one item_key. Stable across machines/Python runs."""
    h = hashlib.sha256(f"{seed}:{item_key}".encode("utf-8")).hexdigest()
    return int(h, 16) % n_folds


@dataclass(frozen=True)
class SplitManifest:
    seed: int
    n_folds: int
    assignment: dict  # item_key (str) -> fold id (int)

    def fold_of(self, item_key: str) -> int:
        return self.assignment[str(item_key)]

    def save(self, path) -> None:
        payload = {"seed": self.seed, "n_folds": self.n_folds, "assignment": self.assignment}
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load(path) -> "SplitManifest":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return SplitManifest(seed=int(d["seed"]), n_folds=int(d["n_folds"]),
                             assignment={str(k): int(v) for k, v in d["assignment"].items()})


def build_manifest(item_keys, n_folds: int = 3, seed: int = 0) -> SplitManifest:
    uniq = sorted(set(str(k) for k in item_keys))
    assignment = {k: item_fold(k, n_folds, seed) for k in uniq}
    return SplitManifest(seed=seed, n_folds=n_folds, assignment=assignment)


def assert_identical(a: SplitManifest, b: SplitManifest) -> None:
    if (a.seed, a.n_folds, a.assignment) != (b.seed, b.n_folds, b.assignment):
        raise AssertionError("SplitManifest mismatch across agents — cross-agent leakage risk")
```
```python
# aide/hygiene/__init__.py
from .manifest import SplitManifest, build_manifest, item_fold, assert_identical

__all__ = ["SplitManifest", "build_manifest", "item_fold", "assert_identical"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest aide/hygiene/tests/test_manifest.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add aide/__init__.py aide/hygiene/__init__.py aide/hygiene/manifest.py aide/hygiene/tests/__init__.py aide/hygiene/tests/test_manifest.py
git commit -q -m "feat(hygiene): cross-agent SplitManifest with deterministic item->fold"
```

---

## Task 2: Item-uniform outer folds

**Files:**
- Create: `aide/hygiene/splits.py`
- Test: `aide/hygiene/tests/test_splits.py`

- [ ] **Step 1: Write the failing test**

```python
# aide/hygiene/tests/test_splits.py
import numpy as np
from aide.hygiene.manifest import build_manifest
from aide.hygiene.splits import outer_folds, row_fold_ids


def test_outer_folds_are_item_disjoint_and_cover_all_items():
    m = build_manifest([f"item{i}" for i in range(30)], n_folds=3, seed=0)
    folds = outer_folds(m)
    assert len(folds) == 3
    for fold in folds:
        assert set(fold.train_item_keys).isdisjoint(fold.oof_item_keys)
    # every item is OOF in exactly one fold
    oof_union = set()
    for fold in folds:
        oof_union |= set(fold.oof_item_keys)
    assert oof_union == set(m.assignment)


def test_row_fold_ids_keep_all_rows_of_an_item_together():
    m = build_manifest(["a", "b"], n_folds=3, seed=0)
    rows = ["a", "a", "a", "b", "b"]  # item a appears 3x, b 2x
    ids = row_fold_ids(rows, m)
    assert ids[0] == ids[1] == ids[2]  # all 'a' rows share a fold
    assert ids[3] == ids[4]            # all 'b' rows share a fold
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest aide/hygiene/tests/test_splits.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aide.hygiene.splits'`.

- [ ] **Step 3: Write minimal implementation**

```python
# aide/hygiene/splits.py
"""Item-uniform OOF folds (and recursive inner folds in Task 3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .manifest import SplitManifest


@dataclass(frozen=True)
class Fold:
    index: int
    train_item_keys: tuple
    oof_item_keys: tuple


def outer_folds(manifest: SplitManifest) -> list:
    items = sorted(manifest.assignment)
    folds = []
    for f in range(manifest.n_folds):
        oof = tuple(k for k in items if manifest.assignment[k] == f)
        trn = tuple(k for k in items if manifest.assignment[k] != f)
        folds.append(Fold(index=f, train_item_keys=trn, oof_item_keys=oof))
    return folds


def row_fold_ids(item_keys_per_row, manifest: SplitManifest) -> np.ndarray:
    return np.array([manifest.assignment[str(k)] for k in item_keys_per_row], dtype=int)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest aide/hygiene/tests/test_splits.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add aide/hygiene/splits.py aide/hygiene/tests/test_splits.py
git commit -q -m "feat(hygiene): item-uniform outer OOF folds"
```

---

## Task 3: Recursive (nested) inner folds for layer-2

**Files:**
- Modify: `aide/hygiene/splits.py` (add `inner_folds`)
- Test: `aide/hygiene/tests/test_splits.py` (append)

- [ ] **Step 1: Write the failing test (append to test_splits.py)**

```python
def test_inner_folds_nest_inside_an_outer_train_set_without_touching_oof():
    from aide.hygiene.splits import outer_folds, inner_folds
    m = build_manifest([f"item{i}" for i in range(30)], n_folds=3, seed=0)
    outer = outer_folds(m)
    o0 = outer[0]
    inner = inner_folds(o0.train_item_keys, n_folds=3, seed=m.seed, outer_index=o0.index)
    # inner folds partition ONLY the outer train items
    inner_items = set()
    for fold in inner:
        inner_items |= set(fold.oof_item_keys)
        assert set(fold.train_item_keys).isdisjoint(fold.oof_item_keys)
    assert inner_items == set(o0.train_item_keys)
    # inner folds NEVER include the outer OOF items (the recursion leakage guard)
    assert inner_items.isdisjoint(o0.oof_item_keys)


def test_inner_folds_are_deterministic_per_outer_index():
    from aide.hygiene.splits import inner_folds
    train = [f"item{i}" for i in range(20)]
    a = inner_folds(train, n_folds=3, seed=0, outer_index=1)
    b = inner_folds(train, n_folds=3, seed=0, outer_index=1)
    assert [f.oof_item_keys for f in a] == [f.oof_item_keys for f in b]
    c = inner_folds(train, n_folds=3, seed=0, outer_index=2)
    # different outer index -> different inner assignment (independent recursion)
    assert [f.oof_item_keys for f in a] != [f.oof_item_keys for f in c]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest aide/hygiene/tests/test_splits.py -q`
Expected: FAIL — `ImportError: cannot import name 'inner_folds'`.

- [ ] **Step 3: Add the implementation to `aide/hygiene/splits.py`**

```python
from .manifest import item_fold  # add to the existing imports at top of splits.py


def inner_folds(train_item_keys, n_folds: int, seed: int, outer_index: int) -> list:
    """Nested OOF over an outer fold's TRAIN items only.

    Used to generate OOF layer-1 predictions that feed the layer-2 stacker so the
    stacker never trains on a member's in-sample (optimistic) predictions. The
    inner seed is derived from (seed, outer_index) so each outer fold recurses
    independently and deterministically.
    """
    items = sorted(set(str(k) for k in train_item_keys))
    inner_seed = seed + 1000 + outer_index
    assign = {k: item_fold(k, n_folds, inner_seed) for k in items}
    folds = []
    for f in range(n_folds):
        oof = tuple(k for k in items if assign[k] == f)
        trn = tuple(k for k in items if assign[k] != f)
        folds.append(Fold(index=f, train_item_keys=trn, oof_item_keys=oof))
    return folds
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest aide/hygiene/tests/test_splits.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add aide/hygiene/splits.py aide/hygiene/tests/test_splits.py
git commit -q -m "feat(hygiene): recursive inner folds for layer-2 OOF"
```

---

## Task 4: Proxy dependency tree

**Files:**
- Create: `aide/hygiene/proxy_tree.py`
- Test: `aide/hygiene/tests/test_proxy_tree.py`

- [ ] **Step 1: Write the failing test**

```python
# aide/hygiene/tests/test_proxy_tree.py
from aide.hygiene.proxy_tree import PROXY_TREE, descendants, all_masked_columns


def test_subject_node_includes_metadata_and_feature_proxies():
    d = descendants("subject")
    assert "subject_key" in d
    assert "meta:family" in d and "meta:macro-family" in d and "meta:parameters" in d
    assert "feat:nn_passrate" in d  # NN/passrate features proxy the subject


def test_benchmark_node_includes_condition_and_data_category():
    d = descendants("benchmark")
    assert "condition" in d          # conditions proxy benchmarks
    assert "data_category" in d
    assert "feat:pool" in d


def test_all_masked_columns_expands_prefixes_atomically():
    cols = ["subject_key", "meta:family", "meta:parameters",
            "feat:nn_passrate__mean", "feat:nn_passrate__max",
            "feat:pool__toklen", "benchmark", "item_emb__0"]
    masked = all_masked_columns(["subject"], cols)
    # every subject-proxy column (incl. prefix-expanded feature groups) is masked
    assert "subject_key" in masked
    assert "meta:family" in masked and "meta:parameters" in masked
    assert "feat:nn_passrate__mean" in masked and "feat:nn_passrate__max" in masked
    # NON-subject columns are NOT masked
    assert "benchmark" not in masked and "item_emb__0" not in masked and "feat:pool__toklen" not in masked
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest aide/hygiene/tests/test_proxy_tree.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aide.hygiene.proxy_tree'`.

- [ ] **Step 3: Write minimal implementation**

```python
# aide/hygiene/proxy_tree.py
"""Proxy dependency tree: fields/feature-groups that proxy subject/benchmark identity.

Dropping an identity node must atomically mask ALL of its descendants, else a proxy
(e.g. model family, or an NN-passrate feature aggregated over the subject) silently
re-leaks the identity the dropout was meant to hide.

Convention: a descendant is either an exact column name (e.g. "subject_key",
"condition") or a feature-group PREFIX (e.g. "feat:nn_passrate", "meta:family") that
matches any column starting with it (so "feat:nn_passrate__mean", "...__max" all mask
together).
"""
from __future__ import annotations

PROXY_TREE = {
    "subject": [
        "subject_key",
        "subject_content",
        "meta:family",
        "meta:macro-family",
        "meta:parameters",
        "meta:organization",
        "meta:release_date",
        "feat:nn_passrate",   # passrate aggregates over the subject
        "feat:subject_mean",  # subject-mean encoding
    ],
    "benchmark": [
        "benchmark",
        "condition",          # conditions proxy benchmarks
        "data_category",
        "feat:pool",          # benchmark-derived pool features
    ],
}


def descendants(node: str) -> list:
    return list(PROXY_TREE.get(node, []))


def all_masked_columns(dropped_nodes, feature_columns) -> set:
    """Expand dropped identity nodes to the concrete set of columns to mask.

    Exact descendant names match exactly; feature-group prefixes match any column
    that equals OR starts with the prefix.
    """
    cols = list(feature_columns)
    masked = set()
    for node in dropped_nodes:
        for proxy in descendants(node):
            for c in cols:
                if c == proxy or c.startswith(proxy):
                    masked.add(c)
    return masked
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest aide/hygiene/tests/test_proxy_tree.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add aide/hygiene/proxy_tree.py aide/hygiene/tests/test_proxy_tree.py
git commit -q -m "feat(hygiene): proxy dependency tree for subject/benchmark identity"
```

---

## Task 5: Entity-keyed proxy dropout

**Files:**
- Create: `aide/hygiene/dropout.py`
- Test: `aide/hygiene/tests/test_dropout.py`

- [ ] **Step 1: Write the failing test**

```python
# aide/hygiene/tests/test_dropout.py
import numpy as np
from aide.hygiene.dropout import apply_proxy_dropout


def _toy():
    cols = ["subject_key", "meta:family", "feat:nn_passrate__mean",
            "benchmark", "condition", "item_emb__0"]
    # 4 rows: subjects s1,s1,s2,s2 ; benchmarks b1,b2,b1,b2
    X = np.ones((4, len(cols)), dtype=np.float32)
    subjects = ["s1", "s1", "s2", "s2"]
    benchmarks = ["b1", "b2", "b1", "b2"]
    return X, cols, subjects, benchmarks


def test_dropping_all_subjects_zeros_every_subject_proxy_column_for_all_rows():
    X, cols, subjects, benchmarks = _toy()
    rng = np.random.default_rng(0)
    Xd, info = apply_proxy_dropout(
        X, cols, subjects=subjects, benchmarks=benchmarks,
        rng=rng, subject_rate=1.0, benchmark_rate=0.0)
    for c in ("subject_key", "meta:family", "feat:nn_passrate__mean"):
        j = cols.index(c)
        assert np.all(Xd[:, j] == 0.0), f"{c} must be fully masked"
    # non-subject columns untouched
    for c in ("benchmark", "condition", "item_emb__0"):
        j = cols.index(c)
        assert np.all(Xd[:, j] == 1.0)
    assert info["dropped_subjects"] == {"s1", "s2"}


def test_dropout_is_entity_consistent_all_rows_of_a_dropped_subject_masked():
    X, cols, subjects, benchmarks = _toy()
    # force-drop only s1 by monkeypatching rate per entity via seed search is brittle;
    # instead assert the invariant: any row whose subject is in dropped_subjects has
    # its subject columns zeroed, and rows of non-dropped subjects keep them.
    rng = np.random.default_rng(3)
    Xd, info = apply_proxy_dropout(
        X, cols, subjects=subjects, benchmarks=benchmarks,
        rng=rng, subject_rate=0.5, benchmark_rate=0.0)
    j = cols.index("subject_key")
    for i, s in enumerate(subjects):
        if s in info["dropped_subjects"]:
            assert Xd[i, j] == 0.0
        else:
            assert Xd[i, j] == 1.0


def test_benchmark_dropout_masks_condition_proxy():
    X, cols, subjects, benchmarks = _toy()
    rng = np.random.default_rng(0)
    Xd, info = apply_proxy_dropout(
        X, cols, subjects=subjects, benchmarks=benchmarks,
        rng=rng, subject_rate=0.0, benchmark_rate=1.0)
    for c in ("benchmark", "condition"):
        j = cols.index(c)
        assert np.all(Xd[:, j] == 0.0)
    assert np.all(Xd[:, cols.index("subject_key")] == 1.0)  # subject side untouched


def test_input_is_not_mutated_in_place():
    X, cols, subjects, benchmarks = _toy()
    X0 = X.copy()
    apply_proxy_dropout(X, cols, subjects=subjects, benchmarks=benchmarks,
                        rng=np.random.default_rng(0), subject_rate=1.0, benchmark_rate=1.0)
    assert np.array_equal(X, X0)  # original untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest aide/hygiene/tests/test_dropout.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aide.hygiene.dropout'`.

- [ ] **Step 3: Write minimal implementation**

```python
# aide/hygiene/dropout.py
"""Aggressive, proxy-aware subject/benchmark dropout.

Entity-keyed: a randomly chosen subset of subjects (and of benchmarks) is "dropped"
per call, and EVERY row of a dropped entity has ALL of that identity's proxy columns
(node + descendants, via proxy_tree) zeroed. This teaches the model to predict on
subjects/benchmarks it has not seen, and — because proxies are masked atomically —
identity cannot leak back through metadata, conditions, or aggregated features.
"""
from __future__ import annotations

import numpy as np

from .proxy_tree import all_masked_columns


def _drop_set(entities, rate: float, rng) -> set:
    uniq = sorted(set(str(e) for e in entities))
    return {e for e in uniq if rng.random() < rate}


def apply_proxy_dropout(X, feature_columns, *, subjects, benchmarks,
                        rng, subject_rate: float, benchmark_rate: float):
    """Return (X_masked, info). Does not mutate X."""
    X = np.asarray(X, dtype=np.float32).copy()
    cols = list(feature_columns)
    col_idx = {c: i for i, c in enumerate(cols)}

    dropped_subj = _drop_set(subjects, subject_rate, rng)
    dropped_bench = _drop_set(benchmarks, benchmark_rate, rng)

    subj_cols = all_masked_columns(["subject"], cols)
    bench_cols = all_masked_columns(["benchmark"], cols)

    subj_arr = np.array([str(s) for s in subjects])
    bench_arr = np.array([str(b) for b in benchmarks])

    if dropped_subj:
        rows = np.isin(subj_arr, list(dropped_subj))
        idx = [col_idx[c] for c in subj_cols if c in col_idx]
        if idx:
            X[np.ix_(rows, idx)] = 0.0
    if dropped_bench:
        rows = np.isin(bench_arr, list(dropped_bench))
        idx = [col_idx[c] for c in bench_cols if c in col_idx]
        if idx:
            X[np.ix_(rows, idx)] = 0.0

    return X, {"dropped_subjects": dropped_subj, "dropped_benchmarks": dropped_bench}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest aide/hygiene/tests/test_dropout.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add aide/hygiene/dropout.py aide/hygiene/tests/test_dropout.py
git commit -q -m "feat(hygiene): entity-keyed proxy-aware subject/benchmark dropout"
```

---

## Task 6: Leakage tripwires

**Files:**
- Create: `aide/hygiene/probes.py`
- Test: `aide/hygiene/tests/test_probes.py`

- [ ] **Step 1: Write the failing test**

```python
# aide/hygiene/tests/test_probes.py
import numpy as np
import pytest
from aide.hygiene.probes import (
    assert_item_disjoint, assert_row_uniform_safe, assert_no_proxy_leak)


def test_assert_item_disjoint_passes_when_disjoint():
    assert_item_disjoint(["a", "b"], ["c", "d"])  # no raise


def test_assert_item_disjoint_raises_on_overlap():
    with pytest.raises(AssertionError):
        assert_item_disjoint(["a", "b", "c"], ["c", "d"])


def test_assert_row_uniform_safe_passes_when_item_stays_in_one_fold():
    rows = ["a", "a", "b"]
    fold_ids = np.array([2, 2, 0])
    assert_row_uniform_safe(rows, fold_ids)  # no raise


def test_assert_row_uniform_safe_raises_when_item_split_across_folds():
    rows = ["a", "a", "b"]
    fold_ids = np.array([2, 1, 0])  # 'a' lands in both fold 2 and fold 1
    with pytest.raises(AssertionError):
        assert_row_uniform_safe(rows, fold_ids)


def test_assert_no_proxy_leak_passes_when_dropped_columns_are_zero():
    cols = ["subject_key", "meta:family", "benchmark"]
    X = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)  # subject proxies zeroed
    assert_no_proxy_leak(X, cols, dropped_nodes=["subject"])  # no raise


def test_assert_no_proxy_leak_raises_when_a_proxy_survives():
    cols = ["subject_key", "meta:family", "benchmark"]
    X = np.array([[0.0, 0.7, 1.0]], dtype=np.float32)  # meta:family survived
    with pytest.raises(AssertionError):
        assert_no_proxy_leak(X, cols, dropped_nodes=["subject"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest aide/hygiene/tests/test_probes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aide.hygiene.probes'`.

- [ ] **Step 3: Write minimal implementation**

```python
# aide/hygiene/probes.py
"""Leakage tripwires. Any harness reporting an NLL should call these first; a failure
aborts the score rather than silently leaking. Adapted from src/oof_folds.py invariants.
"""
from __future__ import annotations

import numpy as np

from .proxy_tree import all_masked_columns


def assert_item_disjoint(train_item_keys, oof_item_keys) -> None:
    overlap = set(str(k) for k in train_item_keys) & set(str(k) for k in oof_item_keys)
    if overlap:
        raise AssertionError(f"item leakage: {len(overlap)} key(s) in both train and oof")


def assert_row_uniform_safe(item_keys_per_row, row_fold_ids) -> None:
    seen = {}
    for k, f in zip((str(x) for x in item_keys_per_row), np.asarray(row_fold_ids).tolist()):
        if k in seen and seen[k] != f:
            raise AssertionError(f"item {k!r} split across folds {seen[k]} and {f}")
        seen[k] = f


def assert_no_proxy_leak(X, feature_columns, dropped_nodes, *, atol: float = 0.0) -> None:
    """Assert every proxy column of a dropped identity node is fully zeroed in X."""
    X = np.asarray(X)
    cols = list(feature_columns)
    masked = all_masked_columns(dropped_nodes, cols)
    for c in masked:
        j = cols.index(c)
        if np.any(np.abs(X[:, j]) > atol):
            raise AssertionError(f"proxy leak: column {c!r} survived dropout of {dropped_nodes}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest aide/hygiene/tests/test_probes.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Update package exports + commit**

Edit `aide/hygiene/__init__.py` to:
```python
from .manifest import SplitManifest, build_manifest, item_fold, assert_identical
from .splits import Fold, outer_folds, inner_folds, row_fold_ids
from .proxy_tree import PROXY_TREE, descendants, all_masked_columns
from .dropout import apply_proxy_dropout
from .probes import assert_item_disjoint, assert_row_uniform_safe, assert_no_proxy_leak

__all__ = [
    "SplitManifest", "build_manifest", "item_fold", "assert_identical",
    "Fold", "outer_folds", "inner_folds", "row_fold_ids",
    "PROXY_TREE", "descendants", "all_masked_columns",
    "apply_proxy_dropout",
    "assert_item_disjoint", "assert_row_uniform_safe", "assert_no_proxy_leak",
]
```

```bash
git add aide/hygiene/probes.py aide/hygiene/tests/test_probes.py aide/hygiene/__init__.py
git commit -q -m "feat(hygiene): leakage tripwires + public API exports"
```

---

## Task 7: Full-suite green + code-review audit

**Files:** none (verification + review).

- [ ] **Step 1: Run the whole hygiene suite**

Run: `python -m pytest aide/hygiene/tests -q`
Expected: PASS (all tasks' tests, ~24 passed). If anything fails, fix before proceeding.

- [ ] **Step 2: Dispatch the code-review agent on the new canon**

Use the `code-review` skill (or general review agent) over the diff of this branch vs `main`, scoped to `aide/`. Focus: hidden leakage paths in the split/dropout logic, determinism, and proxy-tree completeness (does any plausible identity proxy escape the tree?). Save findings to `quality_reports/cross_artifact_aide_hygiene/review.md`.

- [ ] **Step 3: Address any CRITICAL/MAJOR findings, re-run the suite, commit**

```bash
git add -A && git commit -q -m "fix(hygiene): address code-review findings"
```

---

## Self-Review (plan vs spec)

- **§4.1 item-uniform / OOF / recursive / shared manifest** → Tasks 1–3 (manifest determinism + cross-agent identity assert; outer folds item-disjoint; inner recursive folds disjoint from outer OOF). ✓
- **§4.2 proxy-aware subject+benchmark dropout, atomic node+descendants** → Tasks 4–5 (proxy tree + entity-keyed masking; tests assert non-proxy columns survive and proxies don't). ✓
- **§8 leakage tripwire on every score** → Task 6 (`assert_*`), wired into harnesses in Plan 2. ✓
- **§11 autonomous cleanup** → Task 0 (explicit delete list + dangling-ref guard). ✓
- **Code-review audit of surviving canon** → Task 7. ✓
- **Deferred (correct scope):** `src/` member triage → ensemble plan; Drive caching of the manifest/OOF board → Plan 2/4; harness wiring → Plan 2.

No placeholders; signatures consistent across tasks (`Fold`, `apply_proxy_dropout`, `all_masked_columns`, `assert_*` names match their definitions and call sites).
