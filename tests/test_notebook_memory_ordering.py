"""Static assertions that pin the memory-aware ordering in
``notebooks/qwen8b_four_member_stacked.py``.

The notebook trains on a 5M-row split where ``train_ds.item_emb`` is
~80 GB and ``X_train_dense`` is ~25 GB. If both live in memory at once
(plus all the auxiliary state), the Colab A100 high-RAM box (~124 GB)
OOMs.

We enforce two invariants by inspecting the source text:

1. Cell 9 (Model A scoring) must pre-score Model A on TRAIN as well as
   VAL and free ``train_ds`` / ``val_ds`` BEFORE the dense matrix
   build. The code path is ``cache_or_compute("p_a_train", ...)``
   immediately followed by a ``del`` of the dataset names, which is
   what unblocks the OOM observed in the four-member-stacked
   notebook on full-scale training.

2. The calibrator (``_fit_nn_calibrator``) must consume the cached
   ``p_a_train`` rather than re-scoring ``train_ds``. Otherwise the
   80 GB dataset gets rebuilt right after the dense matrices are in
   memory, OOMing again. This invariant is what makes the freed
   datasets actually safe to free.

These tests are static (no notebook execution) so they are cheap to
run on every CI invocation and trip the moment a developer reverts
the ordering -- which previously took a multi-hour Colab run to
notice.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import re
import sys

import pytest

NOTEBOOK_PY = (
    pathlib.Path(__file__).resolve().parents[1]
    / "notebooks"
    / "qwen8b_four_member_stacked.py"
)


@pytest.fixture(scope="module")
def notebook_text() -> str:
    """Read the notebook .py source once per test module."""
    if not NOTEBOOK_PY.exists():
        pytest.skip(f"Notebook source not found: {NOTEBOOK_PY}")
    return NOTEBOOK_PY.read_text(encoding="utf-8")


def test_model_a_train_is_pre_scored_and_cached(notebook_text: str) -> None:
    """Cell 9 must pre-score Model A on train via cache_or_compute."""
    assert 'cache_or_compute(\n    "p_a_train"' in notebook_text or \
        'cache_or_compute("p_a_train"' in notebook_text, (
            "Expected Model A's train predictions to be pre-scored and "
            "cached as 'p_a_train' before the dense matrix build."
        )


def test_train_and_val_ds_are_deleted_after_scoring(notebook_text: str) -> None:
    """After Model A scores both splits, ``train_ds`` / ``val_ds`` are dropped."""
    assert 'for _stale_name in ("train_ds", "val_ds"):' in notebook_text, (
        "Expected an explicit ``del`` of train_ds / val_ds after Model A "
        "scoring; without this the dense matrix build holds ~84 GB of "
        "item embeddings + ~25 GB dense matrix in memory simultaneously."
    )


def test_free_happens_before_dense_matrix_build(notebook_text: str) -> None:
    """The dataset free precedes the dense matrix build textually."""
    free_idx = notebook_text.find('for _stale_name in ("train_ds", "val_ds"):')
    dense_idx = notebook_text.find("X_train_dense = cache_or_compute")
    assert free_idx >= 0, "free block missing"
    assert dense_idx >= 0, "dense matrix build missing"
    assert free_idx < dense_idx, (
        "train_ds / val_ds must be freed BEFORE building X_train_dense; "
        "otherwise peak RAM exceeds the 124 GB Colab box."
    )


def test_calibrator_does_not_rescore_model_a_on_train(notebook_text: str) -> None:
    """The calibrator must reuse the cached ``p_a_train`` instead of re-scoring."""
    # Find the calibrator function block.
    calibrator_match = re.search(
        r"def _fit_nn_calibrator\(\):.*?(?=\n# %%|\nCALIBRATOR_KEY_INPUTS|\Z)",
        notebook_text,
        re.DOTALL,
    )
    assert calibrator_match is not None, "calibrator function not found"
    body = calibrator_match.group(0)
    assert "_score_dataset(train_ds, trained_a)" not in body, (
        "Calibrator must not re-score Model A on train: train_ds was "
        "freed earlier to make memory headroom for the dense matrix."
    )
    assert "p_a_train.astype" in body or "p_a_train_local = p_a_train" in body, (
        "Calibrator must consume the cached ``p_a_train`` rather than "
        "re-scoring."
    )


def test_member2_passes_speed_knobs(notebook_text: str) -> None:
    """The Member 2 cell must thread ``max_bin``, ``force_col_wise``,
    ``log_period``, and ``num_threads`` from CFG into ``fit_gbdt_member``.

    Default LightGBM params take 5-10 min wall-clock on the
    5M x 1200 schema; these knobs cut that to ~1.5-2.5 min while
    staying bit-exact under ``deterministic=True``. A future edit
    that drops them silently regresses training time by 3-5x and
    eats most of a Colab session per re-run.
    """
    assert 'max_bin=int(CFG.get("member2_gbdt", {}).get("max_bin"' in notebook_text
    assert 'force_col_wise=bool(CFG.get("member2_gbdt", {}).get("force_col_wise"' in notebook_text
    assert 'log_period=int(CFG.get("member2_gbdt", {}).get("log_period"' in notebook_text
    # The cache key must include a discriminator so older entries
    # trained with the slow params don't poison a new run.
    assert '"speed_v1"' in notebook_text, (
        "Member 2 cache key must bump when the speed knobs change."
    )
    # The print line must not reference the non-existent
    # ``tree_starts`` attribute (legacy bug; the correct attribute is
    # ``n_trees``).
    assert "gbdt_state.tree_starts" not in notebook_text, (
        "GBDTMemberState exposes ``n_trees``, not ``tree_starts``; "
        "the legacy print line crashes on success."
    )


def test_calibrator_member3_scores_in_chunks(notebook_text: str) -> None:
    """kNN's train scoring must chunk the embedding stack."""
    calibrator_match = re.search(
        r"def _fit_nn_calibrator\(\):.*?(?=\n# %%|\nCALIBRATOR_KEY_INPUTS|\Z)",
        notebook_text,
        re.DOTALL,
    )
    assert calibrator_match is not None
    body = calibrator_match.group(0)
    # The legacy unchunked path materialized the full 80 GB stack in one go.
    bad_pattern = re.compile(
        r"train_item_emb_local\s*=\s*np\.stack\(\s*\[\s*item_emb_lookup\[k\]"
        r"\s+for\s+k\s+in\s+primary\.train\[\"item_key\"\]\]"
    )
    assert not bad_pattern.search(body), (
        "Calibrator must NOT materialize the full train item-embedding "
        "stack at once (~80 GB at 5M rows x 4096 dims). Use chunked "
        "knn_apply_batch instead."
    )
    assert "knn_chunk_rows" in body, (
        "Expected the calibrator to honor a configurable chunk size "
        "via CFG['calibrator']['knn_chunk_rows']."
    )


def _resolve_src_imports(tree: ast.AST) -> dict[str, object]:
    """Return {local_name: imported_object} for every `from src.* import ...`."""
    table: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if not (node.module == "src" or node.module.startswith("src.")):
                continue
            try:
                mod = importlib.import_module(node.module)
            except Exception:
                continue
            for alias in node.names:
                obj = getattr(mod, alias.name, None)
                table[alias.asname or alias.name] = obj
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not (alias.name == "src" or alias.name.startswith("src.")):
                    continue
                try:
                    mod = importlib.import_module(alias.name)
                except Exception:
                    continue
                table[alias.asname or alias.name.split(".")[0]] = mod
    return table


def test_every_src_call_passes_only_declared_kwargs(notebook_text: str) -> None:
    """Static guard: every notebook call to a `from src.* import X` callable
    must use only kwargs declared by X's real ``inspect.signature``.

    This catches the stale-kwarg failure mode (``fit_knn_member(K=...)`` or
    ``knn_apply_batch(item_keys=...)``) on the entire notebook surface, not
    just the few sites we've stumbled into. The check is purely static
    (AST + importlib), so it costs ~1 second of test time and runs every
    CI invocation.
    """
    repo_root = NOTEBOOK_PY.resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    tree = ast.parse(notebook_text)
    imports = _resolve_src_imports(tree)
    if not imports:
        pytest.skip("No src.* imports resolved; cannot run signature audit.")

    issues: list[str] = []
    audited = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        obj = imports.get(name)
        if obj is None:
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        accepts_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if accepts_var_keyword:
            continue
        declared_kw = {
            n for n, p in sig.parameters.items()
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        audited += 1
        for kw in node.keywords:
            if kw.arg is None:
                continue
            if kw.arg not in declared_kw:
                issues.append(
                    f"{name}(...) at notebook line {node.lineno}: "
                    f"unexpected kwarg '{kw.arg}' (declared kwargs: "
                    f"{sorted(declared_kw)})"
                )

    assert not issues, (
        "Found notebook calls passing kwargs the imported src.* callable "
        "does not declare:\n  - " + "\n  - ".join(issues)
    )
    assert audited > 0, "Audit ran but checked zero call sites -- import resolution likely broke."
