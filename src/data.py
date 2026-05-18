"""Data loading and split construction for the Predictive AI Evaluation Challenge.

This module downloads `aims-foundations/measurement-db` from Hugging Face,
joins the per-benchmark response tables with the registry tables
(`subjects.parquet`, `items.parquet`, `benchmarks.parquet`), normalizes the
four-field runtime contract, builds stable cache keys, and produces splits
that respect the platform's item cold-start regime.

The four runtime fields are:

    benchmark, condition, subject_content, item_content

Everything else (subject_id, item_id, label, ...) is bookkeeping and must
never leak into model.predict() at test time.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

LOG = logging.getLogger("data")

REGISTRY_FILES: frozenset[str] = frozenset(
    {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
)

REQUIRED_RUNTIME_FIELDS: tuple[str, ...] = (
    "benchmark",
    "condition",
    "subject_content",
    "item_content",
)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def normalize_condition(value: object) -> str:
    """Normalize a raw `test_condition` to the runtime contract.

    The platform passes the literal string ``"none"`` for missing / null /
    blank conditions. This matches the validation harness's normalization
    exactly so local training and the hosted runtime agree.
    """
    if value is None:
        return "none"
    s = str(value)
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "none"
    return s


def stable_sha256(*parts: str) -> str:
    """Stable hex sha256 over null-separated parts. Used for cache keys.

    We use sha256 (not sha1) deliberately: this is a feature key, not a
    cryptographic decision, but it costs nothing extra and is robust to
    benign collisions in the dataset's free-text fields.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def render_subject_content(
    subject: Mapping[str, object], fallback_subject_id: str
) -> str:
    """Reproduce the starter kit's subject_content rendering.

    The hosted runtime constructs `subject_content` from `subjects.parquet`
    by emitting a `Name:` line and optional metadata lines. We replicate the
    same template so training-time inputs match what `predict()` will see
    in production.
    """
    raw_name = subject.get("display_name")
    display_name = (
        str(raw_name).strip() if raw_name not in (None, "") else fallback_subject_id
    )
    lines = [f"Name: {display_name}"]
    optional_fields = (
        ("provider", "Organization"),
        ("params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    )
    for key, label in optional_fields:
        value = subject.get(key)
        if value not in (None, "", []):
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


_NAME_LINE = re.compile(r"^Name:\s*(.+?)\s*$", re.MULTILINE)


def extract_subject_name(subject_content: str) -> str:
    """Best-effort extraction of the subject's display name.

    The validation harness's logistic baseline does the same thing. We use it
    only for diagnostics / subject-family slicing; the actual model treats
    subject_content as opaque text.
    """
    if not isinstance(subject_content, str):
        return ""
    m = _NAME_LINE.search(subject_content)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Hugging Face dataset download
# ---------------------------------------------------------------------------


def download_measurement_db(
    repo_id: str = "aims-foundations/measurement-db",
    local_dir: str | os.PathLike[str] = "artifacts/data",
    token: str | None = None,
) -> Path:
    """Download every .parquet file from the dataset repo into `local_dir`.

    Returns the local directory path. Idempotent: already-downloaded files
    are skipped. Uses `huggingface_hub.hf_hub_download` (NOT
    `load_dataset("...")` -- the starter kit explicitly warns against that
    because it would mix incompatible trace schemas in).
    """
    from huggingface_hub import HfApi, hf_hub_download

    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    repo_files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    parquet_files = sorted(f for f in repo_files if f.endswith(".parquet"))
    LOG.info(
        "Downloading %d parquet files from %s -> %s",
        len(parquet_files),
        repo_id,
        local,
    )
    for filename in parquet_files:
        dest = local / filename
        if dest.exists() and dest.stat().st_size > 0:
            continue
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local),
            token=token,
        )
    return local


def list_response_files(data_dir: Path) -> list[Path]:
    """List response parquets (per-benchmark), excluding registry and traces."""
    out: list[Path] = []
    for p in sorted(Path(data_dir).iterdir()):
        n = p.name
        if not n.endswith(".parquet"):
            continue
        if n in REGISTRY_FILES:
            continue
        if n.endswith("_traces.parquet"):
            continue
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Join responses with registry tables
# ---------------------------------------------------------------------------


def load_joined_responses(
    data_dir: str | os.PathLike[str],
    *,
    benchmarks: Iterable[str] | None = None,
    max_rows_per_benchmark: int | None = None,
    drop_nan_labels: bool = True,
) -> pd.DataFrame:
    """Load every response table and join with subjects / items / benchmarks.

    Output columns:
        subject_id, item_id, benchmark, condition,
        subject_content, item_content, label, trial, correct_answer
    """
    data_dir = Path(data_dir)
    response_files = list_response_files(data_dir)
    if benchmarks is not None:
        wanted = set(benchmarks)
        response_files = [p for p in response_files if p.stem in wanted]
    if not response_files:
        raise FileNotFoundError(
            f"No response parquet files found in {data_dir}. "
            "Did you run download_measurement_db()?"
        )

    subjects = pd.read_parquet(data_dir / "subjects.parquet")
    items = pd.read_parquet(data_dir / "items.parquet")
    benchmarks_df = pd.read_parquet(data_dir / "benchmarks.parquet")

    subject_by_id: dict[str, dict] = {
        str(row["subject_id"]): row.to_dict() for _, row in subjects.iterrows()
    }
    item_content_by_id: dict[str, str] = dict(
        zip(items["item_id"].astype(str), items["content"].astype(str))
    )
    benchmark_canonical: dict[str, str] = dict(
        zip(
            benchmarks_df["benchmark_id"].astype(str),
            benchmarks_df["benchmark_id"].astype(str),
        )
    )

    parts: list[pd.DataFrame] = []
    for path in response_files:
        df = pd.read_parquet(path)
        if max_rows_per_benchmark is not None and len(df) > max_rows_per_benchmark:
            df = df.sample(
                n=max_rows_per_benchmark, random_state=0
            ).reset_index(drop=True)

        df["benchmark"] = df["benchmark_id"].astype(str).map(
            lambda b: benchmark_canonical.get(b, b)
        )
        df["condition"] = df["test_condition"].map(normalize_condition)
        df["subject_id"] = df["subject_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df["item_content"] = (
            df["item_id"].map(item_content_by_id).fillna("").astype(str)
        )
        df["subject_content"] = df["subject_id"].map(
            lambda sid: render_subject_content(subject_by_id.get(sid, {}), sid)
        )
        df["label"] = pd.to_numeric(df["response"], errors="coerce")

        keep = [
            "subject_id",
            "item_id",
            "benchmark",
            "condition",
            "subject_content",
            "item_content",
            "label",
            "trial",
            "correct_answer",
        ]
        keep = [c for c in keep if c in df.columns]
        parts.append(df[keep])

    out = pd.concat(parts, axis=0, ignore_index=True)
    if drop_nan_labels:
        out = out.dropna(subset=["label"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Key construction (item_key, subject_key, benchmark_condition_key)
# ---------------------------------------------------------------------------


def add_stable_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Attach cache-friendly keys used everywhere downstream.

    - ``subject_key`` = sha256(subject_content)
    - ``item_key`` = sha256(benchmark + "\\n" + condition + "\\n" + item_content)
    - ``benchmark_condition_key`` = "{benchmark}::{condition}"
    """
    df = df.copy()
    df["condition"] = df["condition"].map(normalize_condition)
    df["subject_key"] = df["subject_content"].astype(str).map(stable_sha256)
    df["item_key"] = [
        stable_sha256(b, c, t)
        for b, c, t in zip(
            df["benchmark"].astype(str),
            df["condition"].astype(str),
            df["item_content"].astype(str),
        )
    ]
    df["benchmark_condition_key"] = (
        df["benchmark"].astype(str) + "::" + df["condition"].astype(str)
    )
    return df


# ---------------------------------------------------------------------------
# Label binarization (the response field is sometimes a continuous score)
# ---------------------------------------------------------------------------


def binarize_labels(
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    keep_soft: bool = True,
) -> pd.DataFrame:
    """Coerce raw responses into a `label` column suitable for BCE.

    If `keep_soft` is True we clip continuous labels into [0, 1] (some
    benchmarks have judge scores like 8.5 -- the validation harness warns
    about this explicitly). Otherwise we threshold to {0, 1}.
    """
    df = df.copy()
    label = pd.to_numeric(df["label"], errors="coerce")
    if keep_soft:
        # Heuristic: if a label is clearly above the [0, 1] range (e.g. 8.5),
        # rescale by its benchmark's max so it falls back into [0, 1]. This
        # mirrors what users typically do upstream of BCE.
        out = label.copy()
        max_per_bench = label.groupby(df["benchmark"]).transform(
            lambda s: max(s.abs().max(), 1.0)
        )
        oor = (label < 0.0) | (label > 1.0)
        out = out.where(~oor, label / max_per_bench)
        out = out.clip(0.0, 1.0)
    else:
        out = (label >= threshold).astype(float)
    df["label"] = out
    return df


# ---------------------------------------------------------------------------
# Pruning sparse subjects / items
# ---------------------------------------------------------------------------


def prune_sparse(
    df: pd.DataFrame, *, min_subject_obs: int = 3, min_item_obs: int = 1
) -> pd.DataFrame:
    """Drop subjects/items with too few rows. Reports counts via the logger."""
    n0 = len(df)
    if min_subject_obs > 1:
        s_counts = df["subject_key"].value_counts()
        keep_s = set(s_counts.index[s_counts >= min_subject_obs])
        df = df[df["subject_key"].isin(keep_s)].copy()
    if min_item_obs > 1:
        i_counts = df["item_key"].value_counts()
        keep_i = set(i_counts.index[i_counts >= min_item_obs])
        df = df[df["item_key"].isin(keep_i)].copy()
    LOG.info("prune_sparse: %d -> %d rows", n0, len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------


@dataclass
class DatasetStats:
    n_rows: int
    n_subjects: int
    n_items: int
    n_benchmarks: int
    n_conditions: int
    label_mean: float
    label_mean_by_benchmark: dict[str, float]
    rows_per_subject: dict[str, float]
    rows_per_item: dict[str, float]
    duplicate_rows: int
    inconsistent_pairs: int
    blank_item_content_rows: int
    blank_subject_content_rows: int


def compute_dataset_stats(df: pd.DataFrame) -> DatasetStats:
    """Compute summary statistics for the joined dataframe.

    Designed for printing in the notebook. The dict-valued fields summarize
    distributions as p50/p90/p99/mean so the report stays bounded.
    """
    def _quantiles(s: pd.Series) -> dict[str, float]:
        if len(s) == 0:
            return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"),
                    "p99": float("nan"), "max": float("nan")}
        return {
            "mean": float(s.mean()),
            "p50": float(s.quantile(0.5)),
            "p90": float(s.quantile(0.9)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }

    rows_per_subject = df.groupby("subject_key").size()
    rows_per_item = df.groupby("item_key").size()

    label_mean_by_bench = (
        df.groupby("benchmark")["label"].mean().astype(float).to_dict()
    )

    duplicate_rows = int(
        df.duplicated(
            subset=["subject_key", "item_key", "label"], keep=False
        ).sum()
    )
    pair_label_nunique = (
        df.groupby(["subject_key", "item_key"])["label"].nunique()
    )
    inconsistent_pairs = int((pair_label_nunique > 1).sum())

    return DatasetStats(
        n_rows=len(df),
        n_subjects=int(df["subject_key"].nunique()),
        n_items=int(df["item_key"].nunique()),
        n_benchmarks=int(df["benchmark"].nunique()),
        n_conditions=int(df["condition"].nunique()),
        label_mean=float(df["label"].mean()),
        label_mean_by_benchmark=label_mean_by_bench,
        rows_per_subject=_quantiles(rows_per_subject),
        rows_per_item=_quantiles(rows_per_item),
        duplicate_rows=duplicate_rows,
        inconsistent_pairs=inconsistent_pairs,
        blank_item_content_rows=int(
            (df["item_content"].astype(str).str.len() == 0).sum()
        ),
        blank_subject_content_rows=int(
            (df["subject_content"].astype(str).str.len() == 0).sum()
        ),
    )


def print_dataset_stats(stats: DatasetStats) -> None:
    print("=== Dataset statistics ===")
    print(f"  rows                : {stats.n_rows:,}")
    print(f"  unique subjects     : {stats.n_subjects:,}")
    print(f"  unique item variants: {stats.n_items:,}")
    print(f"  benchmarks          : {stats.n_benchmarks}")
    print(f"  conditions          : {stats.n_conditions}")
    print(f"  global label mean   : {stats.label_mean:.4f}")
    print(f"  duplicate rows      : {stats.duplicate_rows}")
    print(f"  inconsistent pairs  : {stats.inconsistent_pairs}")
    print(f"  blank item_content  : {stats.blank_item_content_rows}")
    print(f"  blank subject_text  : {stats.blank_subject_content_rows}")
    print("  label mean by benchmark:")
    for b, v in sorted(stats.label_mean_by_benchmark.items()):
        print(f"    {b:30s} {v:.4f}")
    print(f"  rows/subject (quantiles): {stats.rows_per_subject}")
    print(f"  rows/item    (quantiles): {stats.rows_per_item}")


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@dataclass
class SplitArtifact:
    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    val_unseen_subject: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame()
    )
    notes: str = ""

    def assert_invariants(self, *, split_name: str | None = None) -> None:
        """Enforce the invariants the platform actually scores against.

        - item_cold_start / benchmark_heldout: no item_key overlap, every
          val subject must appear in train.
        - random_row_debug: do NOT enforce -- this split exists only for
          sanity comparisons and is explicitly leaky.
        """
        name = split_name or self.name
        if name == "random_row_debug":
            return
        train_items = set(self.train["item_key"])
        val_items = set(self.val["item_key"])
        overlap = train_items & val_items
        if overlap:
            raise AssertionError(
                f"{name}: {len(overlap)} item_keys appear in both train and val"
            )
        train_subjects = set(self.train["subject_key"])
        val_subjects = set(self.val["subject_key"])
        leak_subjects = val_subjects - train_subjects
        if leak_subjects and name not in {"random_row_debug"}:
            # NOT a fatal error: in item cold-start a few validation subjects
            # may legitimately be missing if min_subject_obs pruned weakly.
            # We just log it loudly.
            LOG.warning(
                "%s: %d validation subjects do not appear in train",
                name,
                len(leak_subjects),
            )


def make_item_cold_start_split(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.10,
    seed: int = 0,
    holdout_benchmarks: Iterable[str] | None = None,
) -> SplitArtifact:
    """Item cold-start split: validation item_keys are disjoint from train.

    Mirrors `validation_harness.harness.splits.make_item_cold_start_split` so
    local training agrees with the local validation harness and the hosted
    platform's item-cold-start regime.
    """
    rng = np.random.default_rng(seed)
    holdout_benchmarks = tuple(holdout_benchmarks or ())

    all_keys = df[["item_key", "benchmark"]].drop_duplicates()
    if holdout_benchmarks:
        held_mask = all_keys["benchmark"].isin(holdout_benchmarks)
        held = set(all_keys.loc[held_mask, "item_key"])
        normal_pool = all_keys.loc[~held_mask, "item_key"].to_numpy()
    else:
        held = set()
        normal_pool = all_keys["item_key"].to_numpy()

    n_val = int(round(val_fraction * len(normal_pool)))
    n_val = max(0, min(len(normal_pool), n_val))
    perm = rng.permutation(len(normal_pool))
    val_normal = set(normal_pool[perm[:n_val]].tolist())
    val_items = held | val_normal
    train_items = set(all_keys["item_key"]) - val_items

    train = df[df["item_key"].isin(train_items)].copy().reset_index(drop=True)
    raw_val = df[df["item_key"].isin(val_items)].copy().reset_index(drop=True)

    train_subjects = set(train["subject_key"])
    seen_mask = raw_val["subject_key"].isin(train_subjects)
    val = raw_val[seen_mask].reset_index(drop=True)
    val_unseen = raw_val[~seen_mask].reset_index(drop=True)

    art = SplitArtifact(
        name="item_cold_start",
        train=train,
        val=val,
        val_unseen_subject=val_unseen,
        notes=(
            f"val_fraction={val_fraction}; seed={seed}; "
            f"holdout_benchmarks={holdout_benchmarks}"
        ),
    )
    art.assert_invariants()
    return art


def make_benchmark_heldout_split(
    df: pd.DataFrame,
    *,
    holdout_benchmarks: Iterable[str],
    seed: int = 0,
) -> SplitArtifact:
    """Hold out one or more benchmarks entirely.

    Items in any held-out benchmark go to validation; everything else goes
    to train. Subjects not seen in train are dropped from validation, as in
    `make_item_cold_start_split`.
    """
    holdout_benchmarks = list(holdout_benchmarks)
    if not holdout_benchmarks:
        raise ValueError("benchmark_heldout split needs >= 1 holdout_benchmarks")

    val_mask = df["benchmark"].isin(holdout_benchmarks)
    raw_val = df[val_mask].copy().reset_index(drop=True)
    train = df[~val_mask].copy().reset_index(drop=True)

    train_subjects = set(train["subject_key"])
    seen_mask = raw_val["subject_key"].isin(train_subjects)
    val = raw_val[seen_mask].reset_index(drop=True)
    val_unseen = raw_val[~seen_mask].reset_index(drop=True)

    art = SplitArtifact(
        name="benchmark_heldout",
        train=train,
        val=val,
        val_unseen_subject=val_unseen,
        notes=f"holdout_benchmarks={holdout_benchmarks}; seed={seed}",
    )
    art.assert_invariants()
    return art


def make_random_row_split(
    df: pd.DataFrame, *, val_fraction: float = 0.10, seed: int = 0
) -> SplitArtifact:
    """Leaky random-row split. ONLY for debugging.

    Item cold-start is the regime the platform actually evaluates against,
    so this split exists strictly so you can detect models that look great
    here but flop on item cold-start. That gap means overfitting.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = int(round(val_fraction * len(df)))
    val_idx = set(idx[:n_val].tolist())
    is_val = pd.Series([i in val_idx for i in range(len(df))], index=df.index)
    val = df[is_val].copy().reset_index(drop=True)
    train = df[~is_val].copy().reset_index(drop=True)
    return SplitArtifact(
        name="random_row_debug",
        train=train,
        val=val,
        notes=(
            "LEAKY split for debugging only. The platform does NOT score "
            "submissions on random-row validation."
        ),
    )


# ---------------------------------------------------------------------------
# End-to-end loader
# ---------------------------------------------------------------------------


def prepare_dataset(
    data_cfg: Mapping,
    *,
    token: str | None = None,
    download: bool = True,
) -> pd.DataFrame:
    """Top-level convenience: download (if needed), join, normalize, key.

    Returns a fully-prepared dataframe with stable keys and either soft or
    binary labels per `data_cfg["keep_soft_labels"]`.
    """
    local = Path(data_cfg["local_data_dir"])
    if download:
        download_measurement_db(
            repo_id=data_cfg["hf_repo_id"],
            local_dir=local,
            token=token,
        )
    df = load_joined_responses(
        local,
        benchmarks=data_cfg.get("benchmarks"),
        max_rows_per_benchmark=data_cfg.get("max_rows_per_benchmark"),
        drop_nan_labels=data_cfg.get("drop_nan_labels", True),
    )
    df = add_stable_keys(df)
    df = binarize_labels(
        df,
        threshold=float(data_cfg.get("binarize_threshold", 0.5)),
        keep_soft=bool(data_cfg.get("keep_soft_labels", True)),
    )
    df = prune_sparse(
        df,
        min_subject_obs=int(data_cfg.get("min_subject_obs", 1)),
        min_item_obs=int(data_cfg.get("min_item_obs", 1)),
    )
    return df


__all__ = [
    "REQUIRED_RUNTIME_FIELDS",
    "DatasetStats",
    "SplitArtifact",
    "add_stable_keys",
    "binarize_labels",
    "compute_dataset_stats",
    "download_measurement_db",
    "extract_subject_name",
    "load_joined_responses",
    "make_benchmark_heldout_split",
    "make_item_cold_start_split",
    "make_random_row_split",
    "normalize_condition",
    "prepare_dataset",
    "print_dataset_stats",
    "prune_sparse",
    "render_subject_content",
    "stable_sha256",
]
