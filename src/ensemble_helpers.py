"""Helpers for the ensemble-builder Colab notebook.

This module provides the *thin* glue that the notebook needs:

* :func:`extract_bundle`   -- unzip a Codabench bundle into a working dir.
* :func:`load_submodel`    -- import a bundle's ``model.py`` into Python.
* :func:`run_predictions`  -- iterate ``predict()`` over a DataFrame.
* :func:`compute_diversity_metrics` -- pairwise correlations / disagreements
  / Q-statistics between models' predicted probabilities.
* :func:`fit_optimal_weights` -- fit ensemble weights minimising log-loss
  on a held-out set; both simplex (non-negative, sums to 1) and
  unconstrained-logit modes are supported.

The notebook drives these step-by-step so the user can inspect intermediate
artifacts and intervene before committing to a weight configuration.

Why a helpers module instead of inlining everything in the notebook?  The
notebook is meant to be re-run many times (different bundles, different
held-out fractions, ...) and centralising the logic keeps the cells short
and the imports easy to audit.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger("ensemble_helpers")


def _install_pandas_stringdtype_shim() -> bool:
    """Make ``pd.StringDtype('storage', na_value)`` work on older pandas.

    Some Codabench bundles ship a ``preprocessor.pkl`` that was pickled
    with pandas >= 2.3, where ``StringDtype.__init__`` accepts an
    ``na_value`` positional in addition to ``storage``.  On pandas 2.0-2.2
    (still the default in many Colab images as of late-2025) only
    ``storage`` is accepted and unpickling raises::

        TypeError: StringDtype.__init__() takes from 1 to 2 positional
                   arguments but 3 were given

    Rather than forcing every notebook user to upgrade pandas (which can
    cascade into ABI breaks against numpy/pyarrow), we install a one-shot
    shim that swallows the extra positional and pops the ``na_value`` kwarg.
    The result is a ``StringDtype`` instance with the requested ``storage``;
    the ``na_value`` is dropped (pandas <2.3 always uses ``pd.NA``), which
    is a benign no-op for the column-typing the LFM bundle uses.

    Returns ``True`` if the shim was needed and installed, ``False`` if
    the current pandas already accepts the modern signature.
    """
    try:
        from pandas import StringDtype
    except Exception:  # noqa: BLE001
        return False
    # Already shimmed (idempotent): bail.
    if getattr(StringDtype.__init__, "_ensemble_shim", False):
        return True
    # Detect by trying the modern call with na_value=pd.NA.
    try:
        StringDtype("python", pd.NA)  # type: ignore[arg-type]
        return False
    except TypeError:
        pass
    original_init = StringDtype.__init__

    def patched_init(self, storage=None, *args, **kwargs):
        kwargs.pop("na_value", None)
        if args:
            # Silently drop trailing positionals (the na_value from
            # newer-pandas pickles is in args[0]).
            args = ()
        try:
            return original_init(self, storage, *args, **kwargs)
        except TypeError:
            return original_init(self)

    patched_init._ensemble_shim = True  # type: ignore[attr-defined]
    StringDtype.__init__ = patched_init  # type: ignore[method-assign]
    LOG.info(
        "installed pandas StringDtype shim (current pandas=%s)", pd.__version__
    )
    return True


# Install the shim at module import so any pickle-load triggered by
# ``load_submodel`` -> ``model.py`` -> ``load_artifacts`` is safe.
try:
    _install_pandas_stringdtype_shim()
except Exception as _shim_exc:  # noqa: BLE001
    LOG.warning("pandas StringDtype shim install skipped: %s", _shim_exc)


# Per-input field whitelist; mirrors validation_harness/harness/utils.py
INPUT_FIELDS: tuple[str, ...] = (
    "benchmark",
    "condition",
    "subject_content",
    "item_content",
)


# ---------------------------------------------------------------------------
# Bundle I/O
# ---------------------------------------------------------------------------


def extract_bundle(zip_path: str | Path, dest_dir: str | Path) -> Path:
    """Unzip ``zip_path`` into ``dest_dir``; returns the destination directory.

    Idempotent: if ``dest_dir`` already contains a ``model.py`` we skip the
    extract and return it as-is.  Callers that want to force a re-extract
    should delete ``dest_dir`` first.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if (dest_dir / "model.py").exists():
        LOG.info("bundle already extracted at %s (skipping)", dest_dir)
        return dest_dir
    LOG.info("extracting %s -> %s", zip_path.name, dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # ZipFile.extractall handles directory entries that use backslashes
        # on Windows-built archives by translating to '/'.
        zf.extractall(dest_dir)
    # Some archives ship paths with embedded backslashes (e.g. LGAI_fixed_v2);
    # walk and rename if we detect any.
    for p in list(dest_dir.rglob("*")):
        if "\\" in p.name:
            rel = p.relative_to(dest_dir).as_posix().replace("\\", "/")
            new_full = dest_dir / rel
            new_full.parent.mkdir(parents=True, exist_ok=True)
            try:
                p.replace(new_full)
            except Exception:  # noqa: BLE001
                pass
    return dest_dir


def load_submodel(
    submission_dir: str | Path,
    *,
    name: str | None = None,
    reload: bool = False,
) -> ModuleType:
    """Import the ``model.py`` inside ``submission_dir`` as a fresh module.

    Each submission's ``model.py`` defines module-level state (encoder,
    tokenizer, IRT head, ...) so we use a unique synthetic module name to
    keep multiple submissions' state independent.  Re-loading the same
    submission directory (e.g. across notebook reruns) re-imports the file
    from disk so cached state is reset.

    Note: the submission's ``model.py`` calls ``Path(__file__)`` to find its
    own artifacts, so we DO NOT add ``submission_dir`` to ``sys.path``
    (which would risk shadowing other modules).
    """
    submission_dir = Path(submission_dir).resolve()
    model_path = submission_dir / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"model.py not found in {submission_dir}")
    # Reinstall the pandas StringDtype shim in case pandas was upgraded
    # mid-session and the previously-patched class no longer points at the
    # active object (idempotent if not needed).
    try:
        _install_pandas_stringdtype_shim()
    except Exception:  # noqa: BLE001
        pass
    mod_name = name or f"_ensemble_submodel_{abs(hash(str(submission_dir))) & 0xFFFFFFFF:08x}"
    if reload:
        sys.modules.pop(mod_name, None)
    if mod_name in sys.modules and not reload:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    # Make sure relative imports inside model.py (none expected, but cheap to
    # support) and any ``Path(__file__).parent`` lookups resolve to the
    # submission directory.
    spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise AttributeError(f"{model_path} does not define predict()")
    return module


def unload_submodel(module: ModuleType) -> None:
    """Best-effort drop of a submodel's state from sys.modules + GPU memory.

    Submissions hold large encoder weights at module scope; on Colab we
    must free them before loading the next submodel or the second model
    OOMs.  We delete known attribute names, then call ``torch.cuda.empty_cache``.
    """
    drop_attrs = (
        "_ENCODER",
        "_TOKENIZER",
        "_MODEL",
        "_ENCODER_MODEL",
        "_ENCODER_TOKENIZER",
        "JUDGE",
        "_JUDGE",
        "_NN_INDEX",
        "TRAINING_CACHE",
    )
    for attr in drop_attrs:
        if hasattr(module, attr):
            try:
                setattr(module, attr, None)
            except Exception:  # noqa: BLE001
                pass
    name = module.__name__
    sys.modules.pop(name, None)
    try:
        import gc

        gc.collect()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Submission-API runners
# ---------------------------------------------------------------------------


def df_to_inputs(df: pd.DataFrame) -> list[dict[str, str]]:
    """Convert a DataFrame to a list of strict 4-key input dicts.

    Matches ``validation_harness.harness.utils.row_to_input``: only the
    runtime-contract fields are passed through; everything else (label,
    item_id, subject_id, ...) is stripped so we don't accidentally leak
    bookkeeping into the model.
    """
    out: list[dict[str, str]] = []
    for _, row in df.iterrows():
        d: dict[str, str] = {}
        for f in INPUT_FIELDS:
            val = row.get(f, "")
            if val is None:
                val = ""
            d[f] = str(val)
        out.append(d)
    return out


def df_to_labeled(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to the ``labeled`` list shape passed to ``predict``."""
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        d: dict[str, Any] = {f: str(row.get(f, "") or "") for f in INPUT_FIELDS}
        try:
            d["label"] = float(row["label"])
        except Exception:  # noqa: BLE001
            d["label"] = float("nan")
        out.append(d)
    return out


def _resolve_iter(seq, *, use_tqdm: bool, desc: str | None = None):
    """Wrap ``seq`` with ``tqdm`` when ``use_tqdm`` and tqdm is importable.

    Falls back to the bare iterable if tqdm isn't available so the helper
    module stays usable in environments where tqdm wasn't installed.  We
    prefer ``tqdm.auto`` so the bar renders correctly in Jupyter / Colab.
    """
    if not use_tqdm:
        return iter(seq)
    try:
        from tqdm.auto import tqdm

        return tqdm(seq, desc=desc, total=len(seq) if hasattr(seq, "__len__") else None)
    except Exception:  # noqa: BLE001
        return iter(seq)


def run_acquisition_pass(
    labeling_module: ModuleType | None,
    inputs: list[dict[str, str]],
    *,
    log_every: int = 1000,
    use_tqdm: bool = True,
    desc: str | None = "acquisition",
) -> tuple[np.ndarray, str | None]:
    """Score every candidate via ``labeling.acquisition_function``.

    Returns ``(scores, error_reason)``.  On any exception or non-finite
    return value, ``error_reason`` is set and ``scores`` is a uniform-random
    fallback so the caller can still proceed.

    Mirrors ``validation_harness/harness/rounds.py::_safe_acquisition`` but
    accepts ``labeling_module is None`` and treats it as random.

    ``use_tqdm`` adds a per-row progress bar (default).  ``log_every`` still
    emits an INFO-level summary every N rows for log-only consumers.
    """
    n = len(inputs)
    rng = np.random.default_rng(0)
    if labeling_module is None or not hasattr(labeling_module, "acquisition_function"):
        return rng.random(n), "no labeling module / acquisition_function"
    fn = labeling_module.acquisition_function
    scores = np.empty(n, dtype=np.float64)
    t0 = time.time()
    err: str | None = None
    bar = _resolve_iter(range(n), use_tqdm=use_tqdm, desc=desc)
    for i in bar:
        try:
            s = float(fn(inputs[i]))
            if not np.isfinite(s):
                err = f"non-finite at i={i}: {s!r}"
                break
            scores[i] = s
        except Exception as e:  # noqa: BLE001
            err = f"acquisition_function raised at i={i}: {type(e).__name__}: {e}"
            break
        if log_every and (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            LOG.info("  acquisition pass: %d/%d in %.1fs", i + 1, n, elapsed)
    if err is not None:
        # Fall back to random for any unscored items so the caller has a
        # complete vector to work with.
        LOG.warning("acquisition failed (%s); using random fallback", err)
        rng_scores = rng.random(n)
        scores = rng_scores
    LOG.info("acquisition pass: %d items in %.1fs", n, time.time() - t0)
    return scores, err


def select_topk_per_category(
    scores: np.ndarray,
    categories: np.ndarray,
    *,
    k_per_category: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Return the indices selected for labelling (top-K per category).

    Tie-breaking uses an independent random key so ties don't all collapse
    to the same row.  Mirrors the platform's selection logic in
    ``rounds.py``.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    tb = rng.random(n)
    out: list[int] = []
    for cat in np.unique(categories):
        idx = np.flatnonzero(categories == cat)
        order = sorted(
            idx,
            key=lambda i: (-float(scores[i]), float(tb[i])),
        )
        out.extend(order[:k_per_category])
    return np.array(out, dtype=np.int64)


def run_predict_loop(
    model_module: ModuleType,
    inputs: list[dict[str, str]],
    labeled: list[dict[str, Any]] | None,
    *,
    log_every: int = 0,
    label: str = "",
    use_tqdm: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Call ``model_module.predict(input, labeled)`` for every input row.

    Returns the array of predicted probabilities and a small ``stats`` dict
    with timing / fallback counts.  Any exception inside ``predict()`` is
    caught and the row is filled with NaN (so the caller can decide whether
    to treat that as a model failure).

    ``use_tqdm`` adds a per-row progress bar (default).  ``log_every`` still
    emits an INFO-level checkpoint every N rows for log-only consumers (set
    it to 0, the default, when tqdm is doing the visual reporting).
    """
    n = len(inputs)
    preds = np.empty(n, dtype=np.float64)
    n_failed = 0
    n_default = 0
    t0 = time.time()
    desc = f"predict[{label}]" if label else "predict"
    bar = _resolve_iter(range(n), use_tqdm=use_tqdm, desc=desc)
    for i in bar:
        inp = inputs[i]
        try:
            p = model_module.predict(inp, labeled if labeled else None)
            pf = float(p)
        except Exception as e:  # noqa: BLE001
            if n_failed < 5:
                LOG.warning("predict failed at i=%d: %s", i, e)
            n_failed += 1
            pf = float("nan")
        if not np.isfinite(pf):
            n_failed += 1
            pf = float("nan")
        elif abs(pf - 0.5) < 1e-9:
            n_default += 1
        preds[i] = pf
        if log_every and (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            LOG.info("  %s predict: %d/%d in %.1fs", label or "model", i + 1, n, elapsed)
    stats = {
        "n": n,
        "n_failed": n_failed,
        "n_default_05": n_default,
        "seconds": time.time() - t0,
    }
    LOG.info("%s predict done: %s", label or "model", stats)
    return preds, stats


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------


@dataclass
class DiversityReport:
    """Pairwise diversity statistics between models.

    Each field is a square DataFrame indexed by model name.  See
    :func:`compute_diversity_metrics` for definitions.
    """

    pearson: pd.DataFrame
    mean_abs_diff: pd.DataFrame
    kl_div: pd.DataFrame
    q_statistic: pd.DataFrame
    double_fault: pd.DataFrame
    binarized_agreement: pd.DataFrame
    per_model_loss: pd.Series
    per_model_brier: pd.Series

    def diversity_summary(self) -> pd.DataFrame:
        """One-row-per-model summary of mean diversity vs other models."""
        names = list(self.pearson.index)
        rows = []
        for n in names:
            others = [m for m in names if m != n]
            row = {
                "model": n,
                "log_loss": float(self.per_model_loss.get(n, np.nan)),
                "brier": float(self.per_model_brier.get(n, np.nan)),
                "mean_pearson_with_others": float(self.pearson.loc[n, others].mean()),
                "mean_abs_diff_with_others": float(self.mean_abs_diff.loc[n, others].mean()),
                "mean_kl_with_others": float(self.kl_div.loc[n, others].mean()),
                "mean_doublefault_with_others": float(
                    self.double_fault.loc[n, others].mean()
                ),
            }
            rows.append(row)
        return pd.DataFrame(rows).set_index("model").sort_values("log_loss")


def _safe_log(x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return np.log(np.clip(x, eps, 1.0 - eps))


def log_loss_vec(preds: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    """Binary log-loss with NaN-aware masking."""
    p = np.asarray(preds, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mask = np.isfinite(p) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    pp = np.clip(p[mask], eps, 1.0 - eps)
    yy = np.clip(y[mask], 0.0, 1.0)
    return float(-(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp)).mean())


def brier_vec(preds: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(preds, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mask = np.isfinite(p) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    return float(((p[mask] - y[mask]) ** 2).mean())


def compute_diversity_metrics(
    preds_by_model: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    threshold: float = 0.5,
) -> DiversityReport:
    """Compute pairwise diversity stats for a collection of model predictions.

    Definitions used:

    * **Pearson**     -- Pearson correlation of probability vectors.
    * **mean_abs_diff** -- mean over rows of ``|p_a - p_b|``; symmetric.
    * **kl_div**      -- mean Bernoulli KL divergence
                          ``KL(Bern(p_a) || Bern(p_b))``, symmetrised.
    * **q_statistic** -- Yule's Q on the (label != threshold) contingency
                          table; ranges in [-1, 1]; 0 = independent;
                          higher = more agreement.
    * **double_fault** -- ``P(both binarized predictions are wrong)``; a
                          smaller value means the pair is more diverse on
                          the cases where they actually err.
    * **binarized_agreement** -- ``P(binarize(p_a) == binarize(p_b))``.

    All metrics use NaN-aware masking so a partially-failed model doesn't
    corrupt the report.  Returned DataFrames are indexed by model name.
    """
    names = list(preds_by_model.keys())
    M = len(names)
    P = np.stack([preds_by_model[n] for n in names], axis=0)  # (M, N)
    y = np.asarray(labels, dtype=np.float64)
    yb = (y >= threshold).astype(np.int8)

    Pb = (P >= threshold).astype(np.int8)
    finite_each = np.isfinite(P)

    pearson = np.full((M, M), np.nan)
    mad = np.full((M, M), np.nan)
    kl = np.full((M, M), np.nan)
    q = np.full((M, M), np.nan)
    df_ = np.full((M, M), np.nan)
    agree = np.full((M, M), np.nan)

    EPS = 1e-7
    for i in range(M):
        for j in range(M):
            mask = finite_each[i] & finite_each[j] & np.isfinite(y)
            if not mask.any():
                continue
            pa = np.clip(P[i, mask], EPS, 1.0 - EPS)
            pb = np.clip(P[j, mask], EPS, 1.0 - EPS)
            ya = yb[mask]
            ba = Pb[i, mask]
            bb = Pb[j, mask]
            if i == j:
                pearson[i, j] = 1.0
                mad[i, j] = 0.0
                kl[i, j] = 0.0
                q[i, j] = 1.0
                df_[i, j] = float(((ba != ya)).mean())
                agree[i, j] = 1.0
                continue
            # pearson
            if pa.std() > 1e-12 and pb.std() > 1e-12:
                pearson[i, j] = float(np.corrcoef(pa, pb)[0, 1])
            mad[i, j] = float(np.mean(np.abs(pa - pb)))
            # symmetric KL on Bernoulli(p)
            kl_ab = pa * (np.log(pa) - np.log(pb)) + (1.0 - pa) * (
                np.log(1.0 - pa) - np.log(1.0 - pb)
            )
            kl_ba = pb * (np.log(pb) - np.log(pa)) + (1.0 - pb) * (
                np.log(1.0 - pb) - np.log(1.0 - pa)
            )
            kl[i, j] = float(0.5 * (kl_ab.mean() + kl_ba.mean()))
            # Q-statistic on correctness
            ca = (ba == ya).astype(np.int8)
            cb = (bb == ya).astype(np.int8)
            n11 = float(((ca == 1) & (cb == 1)).sum())
            n00 = float(((ca == 0) & (cb == 0)).sum())
            n10 = float(((ca == 1) & (cb == 0)).sum())
            n01 = float(((ca == 0) & (cb == 1)).sum())
            denom = (n11 * n00 + n01 * n10)
            if denom > 0:
                q[i, j] = float((n11 * n00 - n01 * n10) / denom)
            df_[i, j] = float(n00 / mask.sum())
            agree[i, j] = float((ba == bb).mean())

    per_loss = {n: log_loss_vec(P[i], y) for i, n in enumerate(names)}
    per_brier = {n: brier_vec(P[i], y) for i, n in enumerate(names)}

    return DiversityReport(
        pearson=pd.DataFrame(pearson, index=names, columns=names),
        mean_abs_diff=pd.DataFrame(mad, index=names, columns=names),
        kl_div=pd.DataFrame(kl, index=names, columns=names),
        q_statistic=pd.DataFrame(q, index=names, columns=names),
        double_fault=pd.DataFrame(df_, index=names, columns=names),
        binarized_agreement=pd.DataFrame(agree, index=names, columns=names),
        per_model_loss=pd.Series(per_loss),
        per_model_brier=pd.Series(per_brier),
    )


# ---------------------------------------------------------------------------
# Ensemble-weight optimisation
# ---------------------------------------------------------------------------


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable: branch on sign of z so we never call exp() on a
    # large positive number.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``v`` onto the probability simplex."""
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.where(u - cssv / (np.arange(n) + 1) > 0)[0]
    if rho.size == 0:
        return np.full(n, 1.0 / n)
    rho_idx = rho[-1]
    theta = cssv[rho_idx] / (rho_idx + 1)
    return np.maximum(v - theta, 0.0)


@dataclass
class EnsembleFit:
    """Result of a single ensemble-weight optimisation run."""

    model_names: list[str]
    weights: np.ndarray
    method: str
    space: str  # "prob" or "logit"
    log_loss: float
    brier: float
    n_rows: int
    notes: str = ""


def fit_optimal_weights(
    preds_by_model: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    method: str = "simplex_prob",
    init: Sequence[float] | None = None,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> EnsembleFit:
    """Fit ensemble weights minimising log-loss.

    Supported ``method`` values:

    * ``"simplex_prob"`` -- weights >= 0, sum to 1; ensemble is
      ``sum_i w_i p_i``.  Solved by projected gradient on the simplex.
    * ``"simplex_logit"`` -- weights >= 0, sum to 1; ensemble is
      ``sigmoid(sum_i w_i logit(p_i))``.  Same projection-based solver.
    * ``"unconstrained_logit"`` -- weights in R; ensemble is
      ``sigmoid(b + sum_i w_i logit(p_i))``.  Plain logistic regression
      with the model logits as features (closed-form via L-BFGS).
    * ``"uniform_prob"`` -- equal weights in probability space (sanity).
    * ``"uniform_logit"`` -- equal weights in logit space (sanity).
    """
    names = list(preds_by_model.keys())
    P = np.stack([preds_by_model[n] for n in names], axis=0).astype(np.float64)
    y = np.asarray(labels, dtype=np.float64)
    finite_rows = np.all(np.isfinite(P), axis=0) & np.isfinite(y)
    if not finite_rows.any():
        raise ValueError("No fully-finite rows to fit on")
    P = P[:, finite_rows]
    y = y[finite_rows]
    yc = np.clip(y, 0.0, 1.0)
    M, N = P.shape

    if method == "uniform_prob":
        w = np.full(M, 1.0 / M)
        ens = (w[:, None] * P).sum(axis=0)
        return EnsembleFit(
            model_names=names,
            weights=w,
            method=method,
            space="prob",
            log_loss=log_loss_vec(ens, y),
            brier=brier_vec(ens, y),
            n_rows=N,
        )

    if method == "uniform_logit":
        Z = _logit(P)
        w = np.full(M, 1.0 / M)
        ens = _sigmoid((w[:, None] * Z).sum(axis=0))
        return EnsembleFit(
            model_names=names,
            weights=w,
            method=method,
            space="logit",
            log_loss=log_loss_vec(ens, y),
            brier=brier_vec(ens, y),
            n_rows=N,
        )

    if method == "unconstrained_logit":
        Z = _logit(P).T  # (N, M)
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception as e:
            raise RuntimeError(
                "unconstrained_logit needs scikit-learn -- pip install scikit-learn"
            ) from e
        # Use a tiny L2 ridge for numerical stability if some logits are huge.
        clf = LogisticRegression(
            C=1e6,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=max_iter,
        )
        clf.fit(Z, (yc >= 0.5).astype(int), sample_weight=None)
        w = clf.coef_.ravel().astype(np.float64)
        b = float(clf.intercept_.ravel()[0])
        ens = _sigmoid(b + (w[:, None] * _logit(P)).sum(axis=0))
        notes = f"intercept={b:.4f}"
        return EnsembleFit(
            model_names=names,
            weights=w,
            method=method,
            space="logit",
            log_loss=log_loss_vec(ens, y),
            brier=brier_vec(ens, y),
            n_rows=N,
            notes=notes,
        )

    if method in {"simplex_prob", "simplex_logit"}:
        space = "prob" if method == "simplex_prob" else "logit"
        if space == "prob":
            features = P  # (M, N)
        else:
            features = _logit(P)
        w = (
            np.asarray(init, dtype=np.float64)
            if init is not None
            else np.full(M, 1.0 / M)
        )
        w = _project_simplex(w)
        lr = 0.1
        last_loss = float("inf")
        for it in range(max_iter):
            ens_lin = (w[:, None] * features).sum(axis=0)
            if space == "prob":
                ens = np.clip(ens_lin, 1e-7, 1.0 - 1e-7)
                loss = -(yc * np.log(ens) + (1.0 - yc) * np.log(1.0 - ens)).mean()
                # d loss / d w_k = mean over n of (ens - y) / (ens * (1 - ens)) * features[k]
                # Simplifies if we use the logistic form, but here we have probs.
                # Use chain rule with logit reparam:
                grad = ((ens - yc) / (ens * (1.0 - ens)))[None, :] * features
                grad = grad.mean(axis=1)
            else:  # logit space
                ens = _sigmoid(ens_lin)
                loss = -(yc * _safe_log(ens) + (1.0 - yc) * _safe_log(1.0 - ens)).mean()
                grad = ((ens - yc)[None, :] * features).mean(axis=1)
            if loss > last_loss:
                lr *= 0.5
                if lr < 1e-8:
                    break
            else:
                lr = min(lr * 1.05, 0.5)
            w_new = _project_simplex(w - lr * grad)
            if np.linalg.norm(w_new - w) < tol:
                w = w_new
                break
            w = w_new
            last_loss = loss
        # final ensemble
        ens_lin = (w[:, None] * features).sum(axis=0)
        ens = ens_lin if space == "prob" else _sigmoid(ens_lin)
        return EnsembleFit(
            model_names=names,
            weights=w,
            method=method,
            space=space,
            log_loss=log_loss_vec(ens, y),
            brier=brier_vec(ens, y),
            n_rows=N,
        )

    raise ValueError(f"Unknown method: {method!r}")


def apply_weights(
    weights: np.ndarray,
    preds_by_model: Mapping[str, np.ndarray],
    model_names: Sequence[str],
    *,
    space: str = "prob",
    intercept: float = 0.0,
) -> np.ndarray:
    """Apply fitted weights to a fresh set of predictions to get ensemble probs."""
    P = np.stack([preds_by_model[n] for n in model_names], axis=0).astype(np.float64)
    if space == "prob":
        ens = (weights[:, None] * P).sum(axis=0)
        return np.clip(ens, 0.0, 1.0)
    Z = _logit(P)
    return _sigmoid(intercept + (weights[:, None] * Z).sum(axis=0))


__all__ = [
    "INPUT_FIELDS",
    "extract_bundle",
    "load_submodel",
    "unload_submodel",
    "df_to_inputs",
    "df_to_labeled",
    "run_acquisition_pass",
    "select_topk_per_category",
    "run_predict_loop",
    "DiversityReport",
    "compute_diversity_metrics",
    "log_loss_vec",
    "brier_vec",
    "fit_optimal_weights",
    "EnsembleFit",
    "apply_weights",
]
