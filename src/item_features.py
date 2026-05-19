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


__all__ = [
    "POOL_FEATURE_NAMES",
    "apply_zscore",
    "build_feature_matrix",
    "compute_features_for_items",
    "compute_pool_features",
    "feature_names",
    "fit_zscore_stats",
    "load_pool_features",
    "load_zscore_stats",
    "save_pool_features",
    "save_zscore_stats",
]
