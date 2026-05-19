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

    Materializes the full ``[N, D]`` train / val matrices in RAM. For very
    large datasets and high-dimensional encoders (e.g. Qwen3-Embedding-4B
    at d=2560), prefer ``logistic_baseline_on_embeddings_streaming``.
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


def logistic_baseline_on_embeddings_streaming(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    item_emb_lookup: "dict[str, np.ndarray]",
    *,
    label_col: str = "label",
    key_col: str = "item_key",
    batch_size: int = 16384,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    bf16: bool = True,
    device: str | None = None,
    progress: bool = True,
) -> np.ndarray:
    """Memory-safe logistic baseline on item embeddings.

    Trains ``logit = w^T x + b`` minibatch-by-minibatch, looking embeddings up
    from ``item_emb_lookup`` on the fly so the full ``[N, D]`` matrix never
    materializes. This is what you want when ``D`` is large (Qwen3, e5-mistral)
    and ``N`` is on the order of millions.

    Returns clipped val probabilities of shape ``[len(val_df)]``.
    """
    import gc
    import math

    import torch
    from tqdm.auto import tqdm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    first_key = str(train_df[key_col].iloc[0])
    d = int(np.asarray(item_emb_lookup[first_key]).shape[0])

    model = torch.nn.Linear(d, 1).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_keys = train_df[key_col].astype(str).to_numpy()
    train_y = train_df[label_col].to_numpy(dtype=np.float32)

    def _lookup_batch(keys: np.ndarray) -> np.ndarray:
        return np.stack([item_emb_lookup[str(k)] for k in keys]).astype(
            np.float32, copy=False
        )

    n = len(train_y)
    use_amp = bool(bf16 and device == "cuda")

    for epoch in range(epochs):
        perm = np.random.permutation(n)

        iterator = range(0, n, batch_size)
        if progress:
            iterator = tqdm(
                iterator,
                total=math.ceil(n / batch_size),
                desc=f"logistic baseline epoch {epoch + 1}/{epochs}",
                dynamic_ncols=True,
            )

        running_loss = 0.0
        seen = 0
        model.train()

        for start in iterator:
            idx = perm[start : start + batch_size]

            xb_np = _lookup_batch(train_keys[idx])
            yb_np = train_y[idx]

            xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
            yb = torch.from_numpy(yb_np).to(device, non_blocking=True).view(-1, 1)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            bs = len(idx)
            running_loss += float(loss.detach().cpu()) * bs
            seen += bs

            if progress:
                iterator.set_postfix({"loss": f"{running_loss / max(1, seen):.5f}"})

            del xb, yb, logits, loss

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    val_keys = val_df[key_col].astype(str).to_numpy()
    p_val = np.empty(len(val_keys), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        iterator = range(0, len(val_keys), batch_size)
        if progress:
            iterator = tqdm(
                iterator,
                total=math.ceil(len(val_keys) / batch_size),
                desc="logistic baseline predict",
                dynamic_ncols=True,
            )
        for start in iterator:
            end = min(start + batch_size, len(val_keys))
            xb_np = _lookup_batch(val_keys[start:end])
            xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                logits = model(xb).view(-1)
                probs = torch.sigmoid(logits)
            p_val[start:end] = probs.float().cpu().numpy()
            del xb, logits, probs

    return np.clip(p_val, 1e-6, 1.0 - 1e-6)


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


# ---------------------------------------------------------------------------
# Component decomposition + feature-ablation diagnostics (Analysis A / B
# from the design doc; consumed by the notebook's cell 14b)
# ---------------------------------------------------------------------------


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _solo_nll(component: np.ndarray, y: np.ndarray) -> float:
    """Solo NLL: fit sigmoid(a * c + b) on (component, y) and return its NLL.

    Each component gets its own scale and intercept so the metric reflects
    the *information* in ``c``, not its raw magnitude. Falls back to a
    global-mean predictor when the inputs are degenerate.
    """
    c = np.asarray(component, dtype=np.float64).reshape(-1, 1)
    yb = (np.asarray(y, dtype=np.float64) >= 0.5).astype(int)
    if c.size == 0:
        return float("nan")
    if yb.sum() == 0 or (1 - yb).sum() == 0:
        return float("nan")
    if np.allclose(c.std(), 0.0):
        # constant column -- best 2-param logreg is just an intercept
        p = float(np.clip(yb.mean(), 1e-7, 1.0 - 1e-7))
        return float(-(yb * np.log(p) + (1 - yb) * np.log(1.0 - p)).mean())

    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=1e6, max_iter=200, solver="lbfgs")
        clf.fit(c, yb)
        p = clf.predict_proba(c)[:, 1]
        return log_loss(yb.astype(float), p)
    except Exception:
        return float("nan")


def _solo_auc(component: np.ndarray, y: np.ndarray) -> float | None:
    """Solo AUC: scale-invariant ranking quality of ``c`` against ``y``.

    The Solo NLL trick rewards information regardless of scale; Solo AUC
    is a complementary signal that is *only* a ranking measure.
    """
    return auc_roc(np.asarray(y, dtype=float), np.asarray(component, dtype=float))


def component_decomposition_table(
    components: dict[str, np.ndarray],
    y: np.ndarray,
) -> pd.DataFrame:
    """Build the Analysis-B dataframe from a dict of per-row components.

    The expected keys for IRT variants are ``{"irt", "offset", "mlp"}``.
    For the kfactor family the keys are ``{"factor", "mlp"}``. Any extra
    component keys are tolerated and shown alongside.
    """
    rows: list[dict] = []
    for name, c in components.items():
        if not isinstance(c, np.ndarray):
            c = np.asarray(c)
        if c.ndim > 1:
            c = c.reshape(-1)
        rows.append(
            {
                "component": name,
                "var": float(np.var(c)),
                "pearson_y": _pearson_corr(c, y),
                "solo_nll": _solo_nll(c, y),
                "solo_auc": _solo_auc(c, y),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("solo_nll", ascending=True).reset_index(drop=True)


def plot_component_variance(
    components: dict[str, np.ndarray],
    out_path: str | os.PathLike[str],
    *,
    title: str = "Logit-component variance breakdown",
) -> Path:
    """Stacked bar of variance per component (Analysis B sidecar plot)."""
    p = _ensure(out_path)
    names = list(components.keys())
    variances = [float(np.var(c)) for c in components.values()]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(names, variances, color="#4c72b0")
    ax.set_ylabel("Var(component)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=144)
    plt.close(fig)
    return p


# Feature-ablation -----------------------------------------------------------


def _bce_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    """Numerically-stable BCE from raw logits."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).clip(0.0, 1.0)
    # log1pexp(-|z|) + max(-z, 0) is the stable form of log(1 + exp(-z*y_sign))
    # but for soft labels we evaluate p = sigmoid(z) directly with clipping.
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def feature_ablation_table(
    ablations: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    full_key: str = "full",
) -> pd.DataFrame:
    """Build the Analysis-A dataframe from precomputed (p, y) per ablation.

    ``ablations`` maps a label (e.g. "without_pool", "without_alpha",
    "pool_feature[char_len]") to ``(p, y)`` arrays. The row with key
    ``full_key`` is treated as the baseline and Δ-columns are computed
    relative to it.
    """
    rows = []
    for name, (p, y) in ablations.items():
        rows.append(
            {
                "channel_removed": name,
                "val_nll": log_loss(y, p),
                "val_auc": auc_roc(y, p),
            }
        )
    df = pd.DataFrame(rows)
    base = df.loc[df["channel_removed"] == full_key]
    base_nll = float(base["val_nll"].iloc[0]) if not base.empty else float("nan")
    base_auc = base["val_auc"].iloc[0] if not base.empty else None
    base_auc_f = float(base_auc) if base_auc is not None else float("nan")
    df["delta_nll"] = df["val_nll"] - base_nll
    df["delta_auc"] = df["val_auc"].astype(float) - base_auc_f
    return df.sort_values("delta_nll", ascending=False).reset_index(drop=True)


def plot_feature_ablation(
    df: pd.DataFrame,
    out_path: str | os.PathLike[str],
    *,
    title: str = "Feature ablation by Δ NLL",
) -> Path:
    """Horizontal bar of Δ NLL by channel removed (positive = channel matters)."""
    p = _ensure(out_path)
    sub = df.sort_values("delta_nll", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(sub))))
    colors = ["#55a868" if d <= 0 else "#c44e52" for d in sub["delta_nll"]]
    ax.barh(sub["channel_removed"], sub["delta_nll"], color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ NLL (positive = channel is contributing)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=144)
    plt.close(fig)
    return p


__all__ = [
    "MetricBundle",
    "accuracy_at_half",
    "attach_subject_family",
    "auc_roc",
    "bc_mean_with_shrinkage",
    "brier",
    "build_results_dataframe",
    "calibration_table",
    "component_decomposition_table",
    "compute_metrics",
    "expected_calibration_error",
    "feature_ablation_table",
    "global_mean_baseline",
    "log_loss",
    "logistic_baseline_on_embeddings",
    "logistic_baseline_on_embeddings_streaming",
    "metrics_by_group",
    "metrics_by_token_length",
    "plot_calibration_curves",
    "plot_component_variance",
    "plot_feature_ablation",
    "plot_logloss_by_benchmark",
    "plot_perf_vs_token_length",
    "plot_residual_improvement_by_benchmark",
    "plot_val_logloss_by_model",
    "subject_mean_with_shrinkage",
    "token_length_buckets",
]
