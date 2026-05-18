"""Metrics, slices, and plotting for the ablation pipeline.

Primary metric for model selection is **item cold-start validation log-loss**;
everything else is auxiliary.

All plotting functions write to disk and return the path. We never call
plt.show() so the same code path works inside notebooks and scripts.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from .data import extract_subject_name

LOG = logging.getLogger("eval")


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    """Mean BCE. Lower is better. Soft-label safe."""
    y = np.clip(np.asarray(y_true, dtype=float), 0.0, 1.0)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    if y.size == 0:
        return float("nan")
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def brier(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.mean((y - p) ** 2))


def auc_roc(y_true: np.ndarray, p: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    yb = (y >= 0.5).astype(int)
    if yb.sum() == 0 or (1 - yb).sum() == 0:
        return None
    if not np.allclose(y, yb, atol=1e-6):
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = yb == 1
    pos_n = pos.sum()
    neg_n = (~pos).sum()
    if pos_n == 0 or neg_n == 0:
        return None
    return float(
        (ranks[pos].sum() - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n)
    )


def accuracy_at_half(y_true: np.ndarray, p: np.ndarray) -> float:
    yb = (np.asarray(y_true, dtype=float) >= 0.5).astype(int)
    pb = (np.asarray(p, dtype=float) >= 0.5).astype(int)
    if yb.size == 0:
        return float("nan")
    return float((yb == pb).mean())


def expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, *, n_bins: int = 10
) -> float:
    """Equal-width ECE. Lower is better."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += abs(conf - acc) * (mask.sum() / y.size)
    return float(ece)


def calibration_table(
    y_true: np.ndarray, p: np.ndarray, *, n_bins: int = 10
) -> pd.DataFrame:
    """Per-bin probability vs. observed-frequency table."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append(
            {
                "bin": b,
                "lo": float(bins[b]),
                "hi": float(bins[b + 1]),
                "n": int(mask.sum()),
                "mean_pred": float(p[mask].mean()) if mask.any() else float("nan"),
                "mean_obs": float(y[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Slicewise metrics
# ---------------------------------------------------------------------------


@dataclass
class MetricBundle:
    n: int
    log_loss: float
    brier: float
    auc: float | None
    accuracy: float
    ece: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "log_loss": self.log_loss,
            "brier": self.brier,
            "auc": self.auc,
            "accuracy": self.accuracy,
            "ece": self.ece,
        }


def compute_metrics(y: np.ndarray, p: np.ndarray, *, n_bins: int = 10) -> MetricBundle:
    return MetricBundle(
        n=int(y.size),
        log_loss=log_loss(y, p),
        brier=brier(y, p),
        auc=auc_roc(y, p),
        accuracy=accuracy_at_half(y, p),
        ece=expected_calibration_error(y, p, n_bins=n_bins),
    )


def metrics_by_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    pred_col: str = "_pred",
    label_col: str = "label",
    n_bins: int = 10,
    min_n: int = 50,
) -> pd.DataFrame:
    """Per-group metrics. Groups smaller than ``min_n`` are dropped."""
    rows = []
    for key, sub in df.groupby(group_col):
        if len(sub) < min_n:
            continue
        m = compute_metrics(
            sub[label_col].to_numpy(),
            sub[pred_col].to_numpy(),
            n_bins=n_bins,
        )
        d = m.as_dict()
        d[group_col] = key
        rows.append(d)
    cols = [group_col, "n", "log_loss", "brier", "auc", "accuracy", "ece"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("log_loss", ascending=True)


def token_length_buckets(token_lengths: Sequence[int]) -> list[tuple[str, int, int]]:
    """Default bucket scheme: <=128, <=256, <=512, <=1024, >1024."""
    buckets = [
        ("<=128", 0, 128),
        ("<=256", 129, 256),
        ("<=512", 257, 512),
        ("<=1024", 513, 1024),
        (">1024", 1025, 10**9),
    ]
    return buckets


def metrics_by_token_length(
    df: pd.DataFrame,
    token_len_col: str = "item_token_len",
    *,
    pred_col: str = "_pred",
    label_col: str = "label",
    n_bins: int = 10,
) -> pd.DataFrame:
    if token_len_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for label, lo, hi in token_length_buckets(df[token_len_col].tolist()):
        sub = df[(df[token_len_col] >= lo) & (df[token_len_col] <= hi)]
        if len(sub) == 0:
            continue
        m = compute_metrics(
            sub[label_col].to_numpy(), sub[pred_col].to_numpy(), n_bins=n_bins
        )
        d = m.as_dict()
        d["bucket"] = label
        rows.append(d)
    cols = ["bucket", "n", "log_loss", "brier", "auc", "accuracy", "ece"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def attach_subject_family(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `subject_family` column inferred from `subject_content`."""
    df = df.copy()
    df["subject_family"] = (
        df["subject_content"].astype(str).map(extract_subject_name).str.lower()
    )
    df["subject_family"] = df["subject_family"].map(_family_from_name)
    return df


def _family_from_name(name: str) -> str:
    """Cheap heuristic: first word before a hyphen or digit-group."""
    if not name:
        return "<unknown>"
    head = name.split("-")[0].split(":")[0].strip()
    head = "".join(c for c in head if not c.isdigit()).strip()
    return head or "<unknown>"


# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib only, all save to disk and return the Path)
# ---------------------------------------------------------------------------


def _ensure(p: str | os.PathLike[str]) -> Path:
    out = Path(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_val_logloss_by_model(
    results_df: pd.DataFrame,
    out_path: str | os.PathLike[str],
    *,
    metric_col: str = "val_log_loss",
    title: str = "Validation log-loss by model",
) -> Path:
    p = _ensure(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = results_df.groupby("model_name")[metric_col]
    means = grouped.mean()
    stds = grouped.std(ddof=0).fillna(0.0)
    order = means.sort_values().index.tolist()
    xs = np.arange(len(order))
    ax.bar(
        xs,
        means.loc[order].values,
        yerr=stds.loc[order].values,
        capsize=4,
        color="#4c72b0",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("log-loss (lower = better)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=144)
    plt.close(fig)
    return p


def plot_calibration_curves(
    per_run_preds: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str | os.PathLike[str],
    *,
    n_bins: int = 10,
) -> Path:
    p = _ensure(out_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    for name, (y, pred) in per_run_preds.items():
        tab = calibration_table(y, pred, n_bins=n_bins)
        ax.plot(tab["mean_pred"], tab["mean_obs"], "o-", label=name, alpha=0.85)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration curves (item cold-start)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(p, dpi=144)
    plt.close(fig)
    return p


def plot_logloss_by_benchmark(
    per_run_per_bench: dict[str, pd.DataFrame],
    out_path: str | os.PathLike[str],
) -> Path:
    """Bar chart of log-loss per benchmark, grouped by model."""
    p = _ensure(out_path)
    if not per_run_per_bench:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        fig.savefig(p, dpi=144)
        plt.close(fig)
        return p
    benches = sorted(
        set().union(*[set(d["benchmark"]) for d in per_run_per_bench.values()])
    )
    models = list(per_run_per_bench)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(benches)), 5))
    width = 0.8 / max(1, len(models))
    for i, name in enumerate(models):
        d = per_run_per_bench[name].set_index("benchmark").reindex(benches)
        ax.bar(
            np.arange(len(benches)) + i * width,
            d["log_loss"].values,
            width=width,
            label=name,
        )
    ax.set_xticks(np.arange(len(benches)) + width * (len(models) - 1) / 2)
    ax.set_xticklabels(benches, rotation=45, ha="right")
    ax.set_ylabel("log-loss")
    ax.set_title("Log-loss by benchmark (item cold-start)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=144)
    plt.close(fig)
    return p


def plot_residual_improvement_by_benchmark(
    base_df: pd.DataFrame,
    challenger_df: pd.DataFrame,
    out_path: str | os.PathLike[str],
    *,
    base_label: str = "kfactor",
    challenger_label: str = "kfactor_mlp",
) -> Path:
    p = _ensure(out_path)
    merged = base_df.merge(
        challenger_df,
        on="benchmark",
        suffixes=("_base", "_chal"),
        how="inner",
    )
    merged["delta"] = merged["log_loss_base"] - merged["log_loss_chal"]
    merged = merged.sort_values("delta", ascending=True)
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(merged)), 5))
    colors = ["#c44e52" if d < 0 else "#55a868" for d in merged["delta"]]
    ax.bar(np.arange(len(merged)), merged["delta"], color=colors)
    ax.set_xticks(np.arange(len(merged)))
    ax.set_xticklabels(merged["benchmark"], rotation=45, ha="right")
    ax.set_ylabel(f"log-loss({base_label}) - log-loss({challenger_label})")
    ax.set_title(f"{challenger_label} improvement over {base_label} (positive = better)")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(p, dpi=144)
    plt.close(fig)
    return p


def plot_perf_vs_token_length(
    per_run_by_len: dict[str, pd.DataFrame],
    out_path: str | os.PathLike[str],
) -> Path:
    p = _ensure(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, df in per_run_by_len.items():
        if df.empty:
            continue
        ax.plot(df["bucket"], df["log_loss"], "o-", label=name, alpha=0.85)
    ax.set_xlabel("item token length bucket")
    ax.set_ylabel("log-loss")
    ax.set_title("Performance vs. item token length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=144)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Baselines (used by the comparison table)
# ---------------------------------------------------------------------------


def global_mean_baseline(train_labels: np.ndarray, val_labels: np.ndarray) -> np.ndarray:
    """Predict the train-set label mean for every val row."""
    p = float(np.clip(np.mean(train_labels), 1e-6, 1.0 - 1e-6))
    return np.full(val_labels.shape, p, dtype=np.float32)


def subject_mean_with_shrinkage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, *, alpha: float = 20.0
) -> np.ndarray:
    """Per-subject mean shrunk toward the global mean with weight alpha."""
    global_mean = float(np.clip(train_df["label"].mean(), 1e-6, 1.0 - 1e-6))
    grouped = train_df.groupby("subject_key")["label"].agg(["sum", "count"])
    shrunk = (grouped["sum"] + alpha * global_mean) / (grouped["count"] + alpha)
    shrunk = shrunk.clip(1e-6, 1.0 - 1e-6)
    return np.array(
        [shrunk.get(k, global_mean) for k in val_df["subject_key"]], dtype=np.float32
    )


def bc_mean_with_shrinkage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, *, alpha: float = 20.0
) -> np.ndarray:
    """Per-benchmark-condition mean shrunk toward the global mean."""
    global_mean = float(np.clip(train_df["label"].mean(), 1e-6, 1.0 - 1e-6))
    grouped = train_df.groupby("benchmark_condition_key")["label"].agg(
        ["sum", "count"]
    )
    shrunk = (grouped["sum"] + alpha * global_mean) / (grouped["count"] + alpha)
    shrunk = shrunk.clip(1e-6, 1.0 - 1e-6)
    return np.array(
        [shrunk.get(k, global_mean) for k in val_df["benchmark_condition_key"]],
        dtype=np.float32,
    )


def logistic_baseline_on_embeddings(
    item_emb_train: np.ndarray,
    item_emb_val: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 200,
) -> np.ndarray:
    """Logistic regression on raw item embeddings. Returns val probabilities.

    No subject features; this exists as a *lower* bound -- if a k-factor
    model can't beat it, something is wrong.
    """
    from sklearn.linear_model import LogisticRegression

    yb = (y_train >= 0.5).astype(int)
    if yb.sum() == 0 or (1 - yb).sum() == 0:
        return np.full(item_emb_val.shape[0], float(yb.mean()))
    clf = LogisticRegression(
        C=C, max_iter=max_iter, solver="lbfgs", n_jobs=1, verbose=0
    )
    clf.fit(item_emb_train, yb)
    return clf.predict_proba(item_emb_val)[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------
# Comparison dataframe
# ---------------------------------------------------------------------------


def build_results_dataframe(
    rows: Iterable[dict],
    *,
    primary_split: str = "item_cold_start",
    primary_metric: str = "val_log_loss",
) -> pd.DataFrame:
    """Sort by primary metric on the primary split (lower is better)."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    if {"split", primary_metric}.issubset(df.columns):
        df["_is_primary"] = (df["split"] == primary_split).astype(int)
        df = df.sort_values(
            ["_is_primary", primary_metric], ascending=[False, True]
        ).drop(columns=["_is_primary"])
    return df.reset_index(drop=True)


__all__ = [
    "MetricBundle",
    "accuracy_at_half",
    "attach_subject_family",
    "auc_roc",
    "bc_mean_with_shrinkage",
    "brier",
    "build_results_dataframe",
    "calibration_table",
    "compute_metrics",
    "expected_calibration_error",
    "global_mean_baseline",
    "log_loss",
    "logistic_baseline_on_embeddings",
    "metrics_by_group",
    "metrics_by_token_length",
    "plot_calibration_curves",
    "plot_logloss_by_benchmark",
    "plot_perf_vs_token_length",
    "plot_residual_improvement_by_benchmark",
    "plot_val_logloss_by_model",
    "subject_mean_with_shrinkage",
    "token_length_buckets",
]
