"""Offline trainer for the simple logistic baseline.

Reads `train.parquet` (produced by `prepare_split.py`), joins to the
ancillary CSVs (`model_info.csv`, `benchmark_info.csv`), builds a flat
feature table with NO cross-term interactions, fits a binary logistic
regression, and writes:

    pipeline.joblib              fitted sklearn Pipeline
    feature_importance.csv       per-encoded-feature coefficient ranking
    feature_group_importance.csv aggregated by source column / group
    training_report.json         diagnostics

Run from this directory:

    py train_logistic.py \\
        --train-parquet ../../splits/smoke/train.parquet \\
        --val-parquet   ../../splits/smoke/val.parquet
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from features import (
    CATEGORICAL_FEATURES,
    INDICATOR_FEATURES,
    NUMERIC_FEATURES,
    FeatureBuilder,
    coerce_numeric_columns,
    feature_groups,
)

HERE = Path(__file__).resolve().parent


@contextlib.contextmanager
def stage(label: str, total_stages: int, idx_holder: list):
    """Print "[i/N] starting label..." then "  done in Xs (eta Ys)"."""
    idx_holder[0] += 1
    i = idx_holder[0]
    started_at = time.perf_counter()
    started_wall = idx_holder[1]
    print(f"[{i}/{total_stages}] {label} ...", flush=True)
    yield
    dt = time.perf_counter() - started_at
    elapsed_total = time.perf_counter() - started_wall
    avg_per_done = elapsed_total / i
    remaining = max(0, total_stages - i)
    eta_s = avg_per_done * remaining
    print(
        f"     done in {dt:6.2f}s   "
        f"(elapsed {elapsed_total:6.1f}s, eta ~{eta_s:5.1f}s, "
        f"{i}/{total_stages} stages)",
        flush=True,
    )


def build_pipeline(*, n_jobs: int, solver: str) -> Pipeline:
    """ColumnTransformer + LogisticRegression. NO crossed features."""
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=True,
                ),
            ),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(NUMERIC_FEATURES)),
            ("cat", cat_pipe, list(CATEGORICAL_FEATURES)),
            ("ind", "passthrough", list(INDICATOR_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=n_jobs,
    )
    clf = LogisticRegression(
        solver=solver,
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        tol=1e-4,
        verbose=0 if solver == "liblinear" else 1,
    )
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def encoded_feature_names(pipeline: Pipeline) -> list[str]:
    pre: ColumnTransformer = pipeline.named_steps["pre"]
    return list(pre.get_feature_names_out())


def importance_table(pipeline: Pipeline) -> pd.DataFrame:
    clf: LogisticRegression = pipeline.named_steps["clf"]
    coef = clf.coef_.ravel()
    names = encoded_feature_names(pipeline)
    df = pd.DataFrame({"feature": names, "coef": coef})
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return df


def group_importance(per_feature: pd.DataFrame) -> pd.DataFrame:
    groups = feature_groups()
    rows = []
    for group_name, source_cols in groups.items():
        prefixes = []
        for c in source_cols:
            if c in NUMERIC_FEATURES:
                prefixes.append(f"num__{c}")
            elif c in CATEGORICAL_FEATURES:
                prefixes.append(f"cat__{c}_")
            else:
                prefixes.append(f"ind__{c}")
        mask = per_feature["feature"].apply(
            lambda n: any(n == p or n.startswith(p) for p in prefixes)
        )
        sub = per_feature.loc[mask]
        rows.append(
            {
                "group": group_name,
                "n_encoded_features": int(len(sub)),
                "sum_abs_coef": float(sub["abs_coef"].sum()),
                "l2_norm_coef": float(np.sqrt((sub["coef"] ** 2).sum())),
                "max_abs_coef": float(sub["abs_coef"].max()) if len(sub) else 0.0,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("sum_abs_coef", ascending=False)
        .reset_index(drop=True)
    )


def chunked_predict_proba(
    pipeline: Pipeline, X: pd.DataFrame, chunk: int = 200_000, label: str = ""
) -> np.ndarray:
    """Memory-friendly + visible-progress predict_proba for large frames."""
    n = len(X)
    out = np.empty(n, dtype=np.float32)
    iterator = range(0, n, chunk)
    if tqdm is not None and n > chunk:
        iterator = tqdm(iterator, desc=f"predict_proba {label}", total=(n + chunk - 1) // chunk)
    for start in iterator:
        end = min(start + chunk, n)
        out[start:end] = pipeline.predict_proba(X.iloc[start:end])[:, 1].astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-parquet", type=Path, required=True)
    ap.add_argument(
        "--val-parquet",
        type=Path,
        default=None,
        help="Optional: also report dev-set log-likelihood / AUC.",
    )
    ap.add_argument(
        "--model-info-csv",
        type=Path,
        default=HERE / "model_info.csv",
    )
    ap.add_argument(
        "--benchmark-info-csv",
        type=Path,
        default=HERE / "benchmark_info.csv",
    )
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument(
        "--binarize-threshold",
        type=float,
        default=0.5,
        help="Convert continuous labels to {0,1} via label >= threshold.",
    )
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional row cap (uniform random sample) for very fast iteration.",
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel workers for ColumnTransformer (and lbfgs/saga solvers).",
    )
    ap.add_argument(
        "--solver",
        default="lbfgs",
        choices=["liblinear", "lbfgs", "saga"],
        help="liblinear = fast on tiny sparse, single-thread; "
             "lbfgs = good default, parallel BLAS; "
             "saga = best for big sparse with l1/elastic, multi-thread.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(args.n_jobs)
    if "OPENBLAS_NUM_THREADS" not in os.environ:
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.n_jobs)
    if "MKL_NUM_THREADS" not in os.environ:
        os.environ["MKL_NUM_THREADS"] = str(args.n_jobs)

    print(f"BLAS threads = {args.n_jobs}    solver = {args.solver}")
    print(f"Out dir       = {args.out_dir}")
    print(f"Train parquet = {args.train_parquet}")
    if args.val_parquet:
        print(f"Val parquet   = {args.val_parquet}")

    n_stages = 7 if args.val_parquet else 6
    idx_holder = [0, time.perf_counter()]

    with stage("Loading train parquet", n_stages, idx_holder):
        train_df = pd.read_parquet(args.train_parquet)
        if args.max_train_rows and len(train_df) > args.max_train_rows:
            train_df = train_df.sample(
                n=args.max_train_rows, random_state=0
            ).reset_index(drop=True)
        print(f"     {len(train_df):,} rows  (benchmarks={train_df['benchmark'].nunique()})")

    with stage("Building feature lookup tables", n_stages, idx_holder):
        fb = FeatureBuilder.from_csvs(args.model_info_csv, args.benchmark_info_csv)
        print(f"     model_info names: {len(fb.model_info):,}   "
              f"benchmarks: {len(fb.benchmark_info):,}")

    with stage("Vectorizing train features", n_stages, idx_holder):
        X_train = fb.transform_dataframe(train_df, show_progress=True)
        X_train = coerce_numeric_columns(X_train)

    with stage("Preparing labels", n_stages, idx_holder):
        raw_y = train_df["label"].astype(float).values
        y_train = np.clip(raw_y, 0.0, 1.0)
        y_bin = (y_train >= args.binarize_threshold).astype(int)
        n_pos = int(y_bin.sum())
        n_neg = int(len(y_bin) - n_pos)
        frac_clipped = float(((raw_y < 0.0) | (raw_y > 1.0)).mean())
        print(f"     binarize@{args.binarize_threshold}: pos={n_pos:,} "
              f"({n_pos / max(1, len(y_bin)):.2%})   neg={n_neg:,}   "
              f"frac_labels_clipped={frac_clipped:.2%}")

    with stage("Fitting pipeline (this is the long step)", n_stages, idx_holder):
        pipeline = build_pipeline(n_jobs=args.n_jobs, solver=args.solver)
        pipeline.fit(X_train, y_bin)
        n_enc = len(encoded_feature_names(pipeline))
        print(f"     fit converged. encoded feature count = {n_enc}")

    with stage("Train-set scoring", n_stages, idx_holder):
        p_train = chunked_predict_proba(pipeline, X_train, label="(train)")
        train_ll = float(-log_loss(y_bin, np.clip(p_train, 1e-7, 1 - 1e-7)))
        try:
            train_auc = float(roc_auc_score(y_bin, p_train))
        except ValueError:
            train_auc = float("nan")
        print(f"     train mean log-likelihood: {train_ll: .4f}   AUC-ROC: {train_auc:.4f}")

    val_ll = None
    val_auc = None
    if args.val_parquet:
        with stage("Val-set scoring", n_stages, idx_holder):
            val_df = pd.read_parquet(args.val_parquet)
            print(f"     val rows: {len(val_df):,}")
            X_val = coerce_numeric_columns(fb.transform_dataframe(val_df, show_progress=True))
            y_val = np.clip(val_df["label"].astype(float).values, 0.0, 1.0)
            y_val_bin = (y_val >= args.binarize_threshold).astype(int)
            p_val = chunked_predict_proba(pipeline, X_val, label="(val)")
            val_ll = float(-log_loss(y_val_bin, np.clip(p_val, 1e-7, 1 - 1e-7)))
            try:
                val_auc = float(roc_auc_score(y_val_bin, p_val))
            except ValueError:
                val_auc = float("nan")
            print(f"     val mean log-likelihood: {val_ll: .4f}   AUC-ROC: {val_auc:.4f}")

    print("\n" + "=" * 72)
    print(f"Train log-likelihood: {train_ll:8.4f}   AUC-ROC: {train_auc:.4f}")
    if val_ll is not None:
        print(f"Val   log-likelihood: {val_ll:8.4f}   AUC-ROC: {val_auc:.4f}")
    print("=" * 72)

    feat_imp = importance_table(pipeline)
    feat_imp.to_csv(args.out_dir / "feature_importance.csv", index=False)

    grp_imp = group_importance(feat_imp)
    grp_imp.to_csv(args.out_dir / "feature_group_importance.csv", index=False)

    print("\nTop 25 individual features by |coef|:")
    print(feat_imp.head(25).to_string(index=False))
    print("\nFeature-group importance (sum |coef|):")
    print(grp_imp.to_string(index=False))

    pipe_path = args.out_dir / "pipeline.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": list(X_train.columns),
            "binarize_threshold": args.binarize_threshold,
        },
        pipe_path,
        compress=3,
    )
    print(f"\nWrote {pipe_path}")

    report = {
        "n_train_rows": int(len(train_df)),
        "n_features_pre_encoding": int(X_train.shape[1]),
        "n_features_post_encoding": len(encoded_feature_names(pipeline)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "train_mean_log_likelihood": train_ll,
        "train_auc_roc": train_auc,
        "val_mean_log_likelihood": val_ll,
        "val_auc_roc": val_auc,
        "binarize_threshold": args.binarize_threshold,
        "solver": args.solver,
        "n_jobs": args.n_jobs,
        "wall_clock_seconds": round(time.perf_counter() - idx_holder[1], 2),
    }
    (args.out_dir / "training_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()
