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

# Where the version-controlled metadata CSVs (subject + benchmark) live.
# These were lifted from the original ``codabench_submission`` bundle's
# ``model_info.csv`` and ``benchmark_info.csv``: they carry the rich
# structured columns (organization / family / macro_family / parameters
# / release_date for subjects, topic / age / has_conditions for
# benchmarks) that the HF ``subjects.parquet`` / ``benchmarks.parquet``
# either don't have or have mostly null. The metadata-aware model
# variant (``meta_hybrid_irt_kfactor_gated_mlp``) joins on these.
METADATA_DIR: Path = Path(__file__).resolve().parents[1] / "data" / "metadata"
METADATA_MODEL_INFO: Path = METADATA_DIR / "model_info.csv"
METADATA_BENCHMARK_INFO: Path = METADATA_DIR / "benchmark_info.csv"


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
        - full per-(benchmark, condition, content) cache key. Used everywhere
        downstream that wants "this exact row's item identity" (caching,
        embedding deduplication, judge scoring, ...).
    - ``item_split_key`` = sha256(benchmark + "\\n" + content)
        - content-only item identity, ignoring condition. This is the key
        the train/val splitter uses so two rows with the same item_content
        under different conditions ALWAYS land on the same side of the
        split. The encoder sees nearly-identical text for those rows, so a
        condition-aware split (item_key) leaks training items into val
        (~88% on the v1 split) and silently breaks val_NLL-based epoch
        selection.
    - ``benchmark_condition_key`` = "{benchmark}::{condition}"
    """
    df = df.copy()
    df["condition"] = df["condition"].map(normalize_condition)
    df["subject_key"] = df["subject_content"].astype(str).map(stable_sha256)
    bench = df["benchmark"].astype(str)
    cond = df["condition"].astype(str)
    content = df["item_content"].astype(str)
    df["item_key"] = [stable_sha256(b, c, t) for b, c, t in zip(bench, cond, content)]
    df["item_split_key"] = [stable_sha256(b, t) for b, t in zip(bench, content)]
    df["benchmark_condition_key"] = bench + "::" + cond
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

        - item_cold_start / benchmark_heldout: no ``item_split_key``
          overlap (i.e. no shared item content per benchmark), and every
          val subject must appear in train.
        - random_row_debug: do NOT enforce -- this split exists only for
          sanity comparisons and is explicitly leaky.

        We check ``item_split_key`` rather than ``item_key`` because the
        leaderboard tests cold-start by item content; a multi-condition
        benchmark whose same-content rows land on opposite sides of the
        split would pass an ``item_key`` check but still leak ~88% of val
        item content back into train (this is exactly the v1 splits/v1
        regression that broke val-based epoch selection).
        """
        name = split_name or self.name
        if name == "random_row_debug":
            return
        check_col = (
            "item_split_key" if "item_split_key" in self.train.columns else "item_key"
        )
        train_items = set(self.train[check_col])
        val_items = set(self.val[check_col])
        overlap = train_items & val_items
        if overlap:
            raise AssertionError(
                f"{name}: {len(overlap)} {check_col}s appear in both train and val"
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
    val_fraction: float = 0.05,
    seed: int = 0,
    holdout_benchmarks: Iterable[str] | None = None,
    split_key: str = "item_split_key",
) -> SplitArtifact:
    """Item cold-start split: validation items are disjoint from train.

    Splits on ``item_split_key`` (content-only) by default so that two
    rows with the same ``item_content`` under different conditions always
    land on the same side. The legacy behavior (split on ``item_key``,
    which includes condition) leaks ~88% of val item content back into
    train on multi-condition benchmarks; pass ``split_key="item_key"`` to
    opt back into it.

    Mirrors ``validation_harness.harness.splits.make_item_cold_start_split``
    so local training agrees with the validation harness and the hosted
    platform's item-cold-start regime.
    """
    rng = np.random.default_rng(seed)
    holdout_benchmarks = tuple(holdout_benchmarks or ())

    if split_key not in df.columns:
        if split_key == "item_split_key" and {"benchmark", "item_content"}.issubset(
            df.columns
        ):
            df = df.copy()
            bench = df["benchmark"].astype(str)
            content = df["item_content"].astype(str)
            df[split_key] = [stable_sha256(b, t) for b, t in zip(bench, content)]
        else:
            raise KeyError(
                f"{split_key!r} not in df. Call add_stable_keys() first."
            )

    all_keys = df[[split_key, "benchmark"]].drop_duplicates()
    if holdout_benchmarks:
        held_mask = all_keys["benchmark"].isin(holdout_benchmarks)
        held = set(all_keys.loc[held_mask, split_key])
        normal_pool = all_keys.loc[~held_mask, split_key].to_numpy()
    else:
        held = set()
        normal_pool = all_keys[split_key].to_numpy()

    n_val = int(round(val_fraction * len(normal_pool)))
    n_val = max(0, min(len(normal_pool), n_val))
    perm = rng.permutation(len(normal_pool))
    val_normal = set(normal_pool[perm[:n_val]].tolist())
    val_items = held | val_normal
    train_items = set(all_keys[split_key]) - val_items

    train = df[df[split_key].isin(train_items)].copy().reset_index(drop=True)
    raw_val = df[df[split_key].isin(val_items)].copy().reset_index(drop=True)

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
            f"holdout_benchmarks={holdout_benchmarks}; "
            f"split_key={split_key}"
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
    df: pd.DataFrame, *, val_fraction: float = 0.05, seed: int = 0
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


# ---------------------------------------------------------------------------
# Metadata CSV loaders (subject + benchmark structured columns)
# ---------------------------------------------------------------------------


def load_metadata_frames(
    *,
    model_info_path: str | os.PathLike[str] | None = None,
    benchmark_info_path: str | os.PathLike[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the version-controlled metadata CSVs as raw dataframes.

    Defaults to the bundled paths under ``data/metadata/`` so callers
    don't need to know where the files live. Pass explicit paths for
    augmented metadata (e.g. an updated organization map shipped with
    a new model release).

    Returns ``(model_info_df, benchmark_info_df)``. The model variant
    that consumes these (``meta_hybrid_irt_kfactor_gated_mlp``) is the
    sole place where the join semantics are defined, so we keep this
    loader column-agnostic and let
    :class:`src.metadata_features.MetadataPreprocessor` normalize.
    """
    mi_path = Path(model_info_path) if model_info_path else METADATA_MODEL_INFO
    bi_path = Path(benchmark_info_path) if benchmark_info_path else METADATA_BENCHMARK_INFO
    if not mi_path.exists():
        raise FileNotFoundError(
            f"model_info.csv not found at {mi_path}. The metadata model "
            f"variant requires the structured-metadata CSVs under "
            f"data/metadata/ (or pass --model_info_path explicitly)."
        )
    if not bi_path.exists():
        raise FileNotFoundError(
            f"benchmark_info.csv not found at {bi_path}."
        )
    model_info = pd.read_csv(mi_path)
    benchmark_info = pd.read_csv(bi_path)
    LOG.info(
        "Loaded metadata: %d model rows from %s, %d benchmark rows from %s",
        len(model_info),
        mi_path,
        len(benchmark_info),
        bi_path,
    )
    return model_info, benchmark_info


def build_subject_content_lookup(df: pd.DataFrame) -> dict[str, str]:
    """Map ``subject_key`` -> raw ``subject_content`` for the metadata join.

    The metadata preprocessor needs the raw subject_content string so it
    can extract ``display_name`` via the ``Name: ...`` regex. We
    de-duplicate by ``subject_key`` (the sha256 of subject_content)
    since the join is one-to-one on that.
    """
    if "subject_key" not in df.columns:
        raise KeyError(
            "build_subject_content_lookup: df must have a 'subject_key' "
            "column. Call add_stable_keys(df) first."
        )
    if "subject_content" not in df.columns:
        raise KeyError(
            "build_subject_content_lookup: df must have a 'subject_content' "
            "column."
        )
    # ``drop_duplicates`` keeps the first occurrence; subject_key is a
    # deterministic hash of subject_content so any duplicate row has
    # the same subject_content as the first.
    paired = df[["subject_key", "subject_content"]].drop_duplicates(subset=["subject_key"])
    return {
        str(k): str(v)
        for k, v in zip(paired["subject_key"].tolist(), paired["subject_content"].tolist())
    }


def prepare_metadata_artifacts(
    train_df: pd.DataFrame,
    indexer,
    *,
    model_info_df: pd.DataFrame | None = None,
    benchmark_info_df: pd.DataFrame | None = None,
    schema=None,
):
    """One-shot helper: fit MetadataPreprocessor + build per-id tables.

    Returns ``(preprocessor, id_tables)`` ready to attach to a
    :class:`src.models.MetaHybridIRTKFactorGatedMLP` via
    ``model.attach_metadata_tables(id_tables)``.

    Use case (notebook side):

        from src.data import prepare_metadata_artifacts
        from src.models import build_model

        mp, tables = prepare_metadata_artifacts(train_df, indexer)
        model = build_model("meta_hybrid_irt_kfactor_gated_mlp", model_cfg)
        model.attach_metadata_tables(tables)

    The preprocessor object goes into ``runtime_meta.json`` at export
    time so the runtime can reconstruct the same encoding for true
    cold-start subjects/benchmarks.
    """
    from .metadata_features import (
        MetadataPreprocessor,
        MetadataSchema,
        build_metadata_id_tables,
    )

    if model_info_df is None or benchmark_info_df is None:
        mi, bi = load_metadata_frames()
        if model_info_df is None:
            model_info_df = mi
        if benchmark_info_df is None:
            benchmark_info_df = bi

    schema = schema or MetadataSchema()
    mp = MetadataPreprocessor.fit(model_info_df, benchmark_info_df, schema=schema)
    subject_content_by_key = build_subject_content_lookup(train_df)
    tables = build_metadata_id_tables(
        preprocessor=mp,
        subject_to_id=indexer.subject_to_id,
        bc_to_id=indexer.bc_to_id,
        subject_content_by_key=subject_content_by_key,
    )
    return mp, tables


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
