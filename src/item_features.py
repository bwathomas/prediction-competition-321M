"""Pool features computed per unique item.

These are hand-engineered scalar features (token length, has-latex, has-code,
question-count, multiple-choice indicators, language) that the residual MLP
can mix with the item embedding. They live *inside* the MLP because their
interactions with the embedding are not specified a priori; the MLP is the
right place to learn that combination.

The training-time stats (mean/std for z-score normalization) are fit on the
train split only and persisted; val/test reuse the train stats so we don't
leak distribution info from val.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger("item_features")


# Regex helpers --------------------------------------------------------------

# LaTeX: $...$, $$...$$, \(...\), \[...\], or \begin{...}
_RE_LATEX = re.compile(
    r"\$\$.+?\$\$|\$[^\$\n]+?\$|\\\(.+?\\\)|\\\[.+?\\\]|\\begin\{[^}]+\}",
    re.DOTALL,
)
# Code: fenced ```...``` blocks OR 4-space-indented blocks
_RE_CODE_FENCE = re.compile(r"```")
_RE_CODE_INDENT = re.compile(r"(?:^|\n)(?: {4,}|\t)\S", re.MULTILINE)
_RE_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
# Multiple-choice option markers at the start of a line:
#   "A)" / "a)" / "(a)" / "(A)" / "A." / "a."
_RE_MC = re.compile(
    r"(?m)^\s*(?:\([A-Ea-e]\)|[A-Ea-e][\)\.])"
)


POOL_FEATURE_NAMES: tuple[str, ...] = (
    "token_len",
    "char_len",
    "has_latex",
    "has_code",
    "n_questions",
    "n_numbers",
    "is_multiple_choice",
    "n_choices",
    "lang_en",
)


def feature_names() -> list[str]:
    """Return the canonical pool-feature column order."""
    return list(POOL_FEATURE_NAMES)


# Single-item computation ----------------------------------------------------


def _detect_english(text: str) -> float:
    """Return 1.0 if the text looks like English, 0.0 otherwise.

    Tries `langdetect` when installed; falls back to a permissive ASCII /
    English-letter heuristic so the pipeline never crashes if the optional
    dependency is missing. The fallback defaults toward "english" because
    the dataset is overwhelmingly English.
    """
    if not text.strip():
        return 1.0
    try:
        from langdetect import DetectorFactory, detect  # type: ignore

        DetectorFactory.seed = 0
        return 1.0 if detect(text[:2000]) == "en" else 0.0
    except Exception:
        ascii_chars = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        total_alpha = sum(1 for ch in text if ch.isalpha())
        if total_alpha == 0:
            return 1.0
        return 1.0 if ascii_chars / total_alpha >= 0.85 else 0.0


def _token_len_estimate(text: str, tokenizer=None) -> float:
    """Token-length estimate.

    If a HF tokenizer is provided, use it (without specials, no truncation).
    Otherwise fall back to a stable char/4 + 1 proxy that's consistent across
    train and val and doesn't require loading the encoder.
    """
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            return float(len(ids))
        except Exception:
            pass
    return float(len(text) / 4.0 + 1.0)


def compute_pool_features(item_text: str, *, tokenizer=None) -> dict[str, float]:
    """Compute the pool-feature dict for a single item text.

    Stable, side-effect free, deterministic. The output dict's keys match
    :data:`POOL_FEATURE_NAMES` exactly.
    """
    s = str(item_text or "")
    char_len = float(len(s))
    token_len = _token_len_estimate(s, tokenizer=tokenizer)
    has_latex = 1.0 if _RE_LATEX.search(s) else 0.0
    has_code = (
        1.0
        if (_RE_CODE_FENCE.search(s) or _RE_CODE_INDENT.search(s))
        else 0.0
    )
    n_questions = float(s.count("?"))
    n_numbers = float(len(_RE_NUMBER.findall(s)))
    mc_matches = _RE_MC.findall(s)
    n_choices = float(len(mc_matches))
    is_mc = 1.0 if len(mc_matches) >= 2 else 0.0
    lang_en = _detect_english(s)
    return {
        "token_len": float(token_len),
        "char_len": char_len,
        "has_latex": has_latex,
        "has_code": has_code,
        "n_questions": n_questions,
        "n_numbers": n_numbers,
        "is_multiple_choice": is_mc,
        "n_choices": n_choices,
        "lang_en": lang_en,
    }


# Batched computation across an item dataframe -------------------------------


def compute_features_for_items(
    item_df: pd.DataFrame,
    *,
    text_col: str = "item_content",
    key_col: str = "item_key",
    tokenizer=None,
    progress: bool = False,
) -> pd.DataFrame:
    """Compute pool features for every row of ``item_df``.

    ``item_df`` should already be deduplicated by ``key_col``. Returns a new
    dataframe with ``key_col`` plus :data:`POOL_FEATURE_NAMES` columns.
    """
    iterator: Iterable = item_df[[key_col, text_col]].itertuples(index=False)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                iterator,
                total=len(item_df),
                desc="Pool features",
                dynamic_ncols=True,
            )
        except Exception:
            pass

    rows: list[dict] = []
    for tup in iterator:
        key, text = tup[0], tup[1]
        feats = compute_pool_features(str(text or ""), tokenizer=tokenizer)
        feats[key_col] = str(key)
        rows.append(feats)
    df = pd.DataFrame(rows)
    cols = [key_col] + list(POOL_FEATURE_NAMES)
    return df[cols].reset_index(drop=True)


# Z-score stats --------------------------------------------------------------


def fit_zscore_stats(
    train_features: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Fit per-feature mean/std from a *train-only* features dataframe."""
    cols = list(feature_cols or POOL_FEATURE_NAMES)
    stats: dict[str, dict[str, float]] = {}
    for c in cols:
        v = train_features[c].astype(float).to_numpy()
        mean = float(np.mean(v)) if v.size else 0.0
        std = float(np.std(v)) if v.size else 1.0
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        stats[c] = {"mean": mean, "std": std}
    return stats


def apply_zscore(
    features: pd.DataFrame,
    stats: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Apply z-score normalization in-place-safe (returns a copy)."""
    out = features.copy()
    for c, s in stats.items():
        if c in out.columns:
            out[c] = (out[c].astype(float) - s["mean"]) / max(1e-6, s["std"])
    return out


# Caching helpers ------------------------------------------------------------


def save_pool_features(features: pd.DataFrame, cache_path: Path) -> Path:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cache_path, index=False)
    return cache_path


def load_pool_features(cache_path: Path) -> pd.DataFrame | None:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        return pd.read_parquet(cache_path)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Failed to load %s (%s); recomputing", cache_path, exc)
        return None


def save_zscore_stats(stats: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return path


def load_zscore_stats(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Failed to load %s (%s)", path, exc)
        return None


# Materialize aligned feature matrix -----------------------------------------


def build_feature_matrix(
    item_keys: Sequence[str],
    features_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str] | None = None,
    key_col: str = "item_key",
) -> np.ndarray:
    """Return a ``[len(item_keys), n_feat]`` float32 matrix.

    Missing keys are filled with zeros (after z-scoring this is the mean,
    which is a safe fallback).
    """
    cols = list(feature_cols or POOL_FEATURE_NAMES)
    n_feat = len(cols)
    lookup = features_df.set_index(key_col)
    # Reindex to align with item_keys; missing rows become NaN, then 0.
    aligned = lookup.reindex([str(k) for k in item_keys])[cols]
    arr = aligned.to_numpy(dtype=np.float32, copy=True)
    arr[~np.isfinite(arr)] = 0.0
    return arr


# Centroid-distance features ------------------------------------------------
#
# These are *embedding-derived* scalar features (not text-derived like
# the canonical 9 above): the squared L2 distances from each item's
# embedding to the top-m nearest k-means centroids, sorted ascending.
# They give the residual MLP a soft cluster signal -- "this item is
# very close to centroid 12 (typical short MCQ math) and somewhat close
# to centroid 7 (Latex-heavy)" -- without committing to a single hard
# id like the existing ``cluster_ids`` channel does.
#
# We surface these through the same plumbing as the text features:
# they're appended as extra columns of the pool-feature dataframe,
# mean/std-fit against the train split, and concatenated into the
# residual MLP via ModelConfig.pool_feature_dim. The runtime computes
# them from the cached centroids file + the per-prediction item
# embedding, so no additional artifact beyond cluster_centroids.npy
# needs to ship.


def centroid_distance_feature_names(top_m: int) -> list[str]:
    """Canonical ``centroid_dist_0`` ... ``centroid_dist_{top_m-1}`` order.

    The numbering is the rank of the centroid (0 = nearest), NOT the
    centroid id, since centroid ids are not stable across runs (kmeans
    is initialization-dependent) but the rank ordering of
    "1st nearest vs 2nd nearest" is what the model actually uses.
    """
    if int(top_m) <= 0:
        raise ValueError(f"top_m must be > 0; got {top_m}")
    return [f"centroid_dist_{i}" for i in range(int(top_m))]


def build_centroid_distance_features(
    item_keys: Sequence[str],
    item_emb_lookup: dict[str, np.ndarray] | dict[str, list[float]],
    centroids: np.ndarray,
    *,
    top_m: int,
) -> pd.DataFrame:
    """Compute per-item top-m centroid-distance columns.

    Returns a DataFrame with columns
    ``["item_key", "centroid_dist_0", ..., "centroid_dist_{top_m-1}"]``,
    one row per ``item_keys`` entry, sorted by ascending squared L2
    distance. Distances are squared L2 (matches what FAISS' IndexFlatL2
    and our :func:`src.clustering.compute_top_m_distances` emit) so the
    runtime / training paths stay numerically identical.

    Items missing from ``item_emb_lookup`` get the sentinel value
    ``np.nan`` -- the consumer is expected to z-score these (NaN -> 0
    after z-score), which is the "treat unknown as average" fallback the
    rest of the pool-feature pipeline already uses.
    """
    from .clustering import compute_top_m_distances

    if int(top_m) <= 0:
        raise ValueError(f"top_m must be > 0; got {top_m}")

    keys = [str(k) for k in item_keys]

    # Two passes so we never pay a stack/copy for items that are
    # missing from the lookup; tracked via a bool mask.
    present_idx: list[int] = []
    present_vecs: list[np.ndarray] = []
    for i, k in enumerate(keys):
        v = item_emb_lookup.get(k)
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(
                f"item_emb_lookup[{k!r}] must be 1-D; got shape {arr.shape}"
            )
        present_idx.append(i)
        present_vecs.append(arr)

    cols = centroid_distance_feature_names(int(top_m))
    if not present_vecs:
        # No items had embeddings -- emit an all-NaN frame so caller can
        # detect and fall back. (Shape still consistent with cols.)
        empty = pd.DataFrame(
            {c: np.full(len(keys), np.nan, dtype=np.float32) for c in cols}
        )
        empty.insert(0, "item_key", keys)
        return empty

    X = np.stack(present_vecs, axis=0)
    if X.shape[1] != int(centroids.shape[1]):
        raise ValueError(
            f"item embedding dim {X.shape[1]} != centroid dim {int(centroids.shape[1])}"
        )

    _, dists = compute_top_m_distances(
        centroids,
        X,
        top_m=int(top_m),
    )

    out = np.full((len(keys), int(top_m)), np.nan, dtype=np.float32)
    for row_pos, idx in enumerate(present_idx):
        out[idx] = dists[row_pos]

    df = pd.DataFrame({c: out[:, i] for i, c in enumerate(cols)})
    df.insert(0, "item_key", keys)
    return df


def merge_pool_and_centroid_features(
    pool_df: pd.DataFrame,
    centroid_df: pd.DataFrame | None,
    *,
    key_col: str = "item_key",
) -> tuple[pd.DataFrame, list[str]]:
    """Outer-join ``pool_df`` with ``centroid_df`` on ``key_col``.

    Returns ``(merged_df, feature_cols)`` where ``feature_cols`` is the
    canonical column order the consumer should pass into
    :func:`fit_zscore_stats` / :func:`build_feature_matrix`.

    When ``centroid_df`` is ``None`` we just return the pool features
    plus the canonical 9-column order. This is the "centroid distances
    disabled" path so callers can use one merge call for both modes.
    """
    base_cols = list(POOL_FEATURE_NAMES)
    if centroid_df is None or len(centroid_df.columns) <= 1:
        return pool_df, base_cols

    # The centroid_df has [item_key, centroid_dist_0, ..., centroid_dist_{m-1}].
    extra_cols = [c for c in centroid_df.columns if c != key_col]
    merged = pool_df.merge(
        centroid_df[[key_col, *extra_cols]],
        on=key_col,
        how="left",
        validate="one_to_one",
    )
    return merged, base_cols + extra_cols


# CoT / item-type interaction helpers ---------------------------------------
#
# These power the "secondary" dense members (M2/M6/M8) where we want
# explicit condition x item-form interactions, e.g. "does chain-of-thought
# help THIS kind of item". They are deliberately kept OUT of the canonical
# POOL_FEATURE_NAMES so the primary model (M1) -- whose cached input width is
# baked into its checkpoint -- is unaffected. Item-type is per-item; the CoT
# crosses are per-ROW (condition lives on the row, not the item).

ITEM_TYPE_NAMES: tuple[str, ...] = (
    "type_mcq",
    "type_code",
    "type_math",
    "type_prose",
)

# Tokens that mark a chain-of-thought / step-by-step style condition.
_COT_TOKENS: tuple[str, ...] = (
    "cot",
    "chain-of-thought",
    "chain of thought",
    "chainofthought",
    "step by step",
    "step-by-step",
    "scratchpad",
    "reasoning",
)

# Base features the CoT flag is crossed with (chosen to avoid collinear /
# low-signal crosses: char_len ~ token_len, n_questions/lang_en weak).
COT_INTERACTION_BASE: tuple[str, ...] = (
    "has_code",
    "has_latex",
    "is_multiple_choice",
    "n_numbers",
    "token_len",
    "type_mcq",
    "type_code",
    "type_math",
)


def is_cot_from_condition(condition: str) -> float:
    """1.0 if the (normalized) condition looks like a CoT/step-by-step mode."""
    s = str(condition or "").strip().lower()
    if not s or s == "none":
        return 0.0
    return 1.0 if any(tok in s for tok in _COT_TOKENS) else 0.0


def item_type_onehot(pool: dict) -> dict[str, float]:
    """Coarse, mutually-exclusive item type from the form features.

    Priority order code > mcq > math > prose, so every item lands in exactly
    one bucket. ``pool`` is a dict as returned by :func:`compute_pool_features`.
    """
    has_code = float(pool.get("has_code", 0.0)) >= 0.5
    is_mc = float(pool.get("is_multiple_choice", 0.0)) >= 0.5
    has_latex = float(pool.get("has_latex", 0.0)) >= 0.5
    n_numbers = float(pool.get("n_numbers", 0.0))
    is_math = has_latex or n_numbers >= 5.0
    t = {n: 0.0 for n in ITEM_TYPE_NAMES}
    if has_code:
        t["type_code"] = 1.0
    elif is_mc:
        t["type_mcq"] = 1.0
    elif is_math:
        t["type_math"] = 1.0
    else:
        t["type_prose"] = 1.0
    return t


def cot_interaction_names(bases: Sequence[str] = COT_INTERACTION_BASE) -> list[str]:
    """Canonical ``cot_x_<base>`` column order."""
    return [f"cot_x_{b}" for b in bases]


def build_cot_interactions(
    base: np.ndarray,
    base_names: Sequence[str],
    is_cot: np.ndarray,
    *,
    bases: Sequence[str] = COT_INTERACTION_BASE,
) -> tuple[np.ndarray, list[str]]:
    """Build the ``[N, len(bases)]`` CoT-interaction matrix.

    ``base`` is a ``[N, F]`` per-row matrix (z-scored pool features + item-type
    one-hots), ``base_names`` its column names, ``is_cot`` a ``[N]`` 0/1 vector.
    Each output column is ``is_cot * base[:, base_col]`` -- i.e. it is zero on
    non-CoT rows and equal to the (z-scored) base value on CoT rows, which lets
    a linear/MLP head learn a CoT-specific slope per base feature.
    """
    name_to_col = {str(n): i for i, n in enumerate(base_names)}
    isc = np.asarray(is_cot, dtype=np.float32).reshape(-1, 1)
    cols: list[np.ndarray] = []
    out_names: list[str] = []
    for b in bases:
        j = name_to_col.get(str(b))
        if j is None:
            continue
        cols.append(base[:, j : j + 1].astype(np.float32) * isc)
        out_names.append(f"cot_x_{b}")
    if not cols:
        return np.zeros((int(base.shape[0]), 0), dtype=np.float32), []
    return np.concatenate(cols, axis=1).astype(np.float32), out_names


__all__ = [
    "COT_INTERACTION_BASE",
    "ITEM_TYPE_NAMES",
    "POOL_FEATURE_NAMES",
    "apply_zscore",
    "build_centroid_distance_features",
    "build_cot_interactions",
    "build_feature_matrix",
    "centroid_distance_feature_names",
    "compute_features_for_items",
    "compute_pool_features",
    "cot_interaction_names",
    "feature_names",
    "fit_zscore_stats",
    "is_cot_from_condition",
    "item_type_onehot",
    "load_pool_features",
    "load_zscore_stats",
    "merge_pool_and_centroid_features",
    "save_pool_features",
    "save_zscore_stats",
]
