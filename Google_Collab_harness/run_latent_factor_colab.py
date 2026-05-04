"""End-to-end runner for the metadata-only latent-factor model.

This is the single command Colab calls. It:

1. Loads the joined responses parquet (from prepare_split.py output).
2. Fits the metadata preprocessor on TRAIN ONLY.
3. Aggregates train rows into (model, benchmark, condition) cells (since
   the model ignores item_content, this is loss-equivalent).
4. Trains the latent-factor model with AdamW + AMP + early stopping.
5. Reports row-level full-val log-likelihood.
6. Builds a submission folder and runs the existing official-like
   validation harness across 3 seeds.
7. Writes:
     outputs/latent_factor/best_model.pt
     outputs/latent_factor/preprocessor.pkl
     outputs/latent_factor/metrics.json
     outputs/latent_factor/runs.csv
     outputs/latent_factor/baseline_comparison.csv

Optional `--sweep` runs a small grid (latent_dim x weight_decay x dropout)
and picks the best by full-val log-likelihood.

Example (Colab):
    python run_latent_factor_colab.py \\
        --data-dir starting_kit/Data \\
        --splits-dir validation_harness/splits/v1 \\
        --model-info-csv starting_kit/Model_Info/model_info.csv \\
        --benchmark-info-csv starting_kit/benchmark_info/benchmark_info.csv \\
        --validation-harness-dir validation_harness \\
        --output-dir outputs/latent_factor \\
        --latent-dim 16 --epochs 30 --batch-size 65536 --amp --aggregate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from latent_factor_pytorch import (
    LFConfig,
    LatentFactorModel,
    MetadataDataset,
    MetadataPreprocessor,
    aggregate_cells,
    detect_label_column,
    evaluate,
    export_metadata_lookup,
    save_artifacts,
    train_model,
)


def _setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_env(device: torch.device) -> None:
    print(f"torch={torch.__version__}  cuda={torch.version.cuda}  device={device}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        print(f"GPU: {torch.cuda.get_device_name(idx)}")
        try:
            free, total = torch.cuda.mem_get_info(idx)
            print(f"GPU memory: free={free/1e9:.2f}GB total={total/1e9:.2f}GB")
        except Exception:
            pass


def _ensure_split(args: argparse.Namespace) -> tuple[Path, Path]:
    """Build the official-like split if --splits-dir is empty."""
    splits_dir = Path(args.splits_dir)
    train_p = splits_dir / "train.parquet"
    val_p = splits_dir / "val.parquet"
    if train_p.exists() and val_p.exists():
        return train_p, val_p

    if not args.data_dir:
        raise SystemExit(
            f"Splits dir {splits_dir} doesn't contain train.parquet/val.parquet, "
            "and --data-dir was not provided to build a fresh split."
        )

    splits_dir.mkdir(parents=True, exist_ok=True)
    harness_dir = Path(args.validation_harness_dir).resolve()
    cmd = [
        sys.executable,
        str(harness_dir / "scripts" / "prepare_split.py"),
        "--data-dir", str(Path(args.data_dir).resolve()),
        "--out-dir", str(splits_dir.resolve()),
        "--val-fraction", str(args.val_fraction),
        "--seed", str(args.split_seed),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    return train_p, val_p


def _build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pp: MetadataPreprocessor,
    label_col: str,
    aggregate: bool,
) -> tuple[MetadataDataset, MetadataDataset, dict]:
    info = {}
    if aggregate:
        agg = aggregate_cells(train_df, label_col=label_col)
        train_targets = torch.from_numpy(agg["target"].to_numpy(dtype=np.float32))
        train_weights = torch.from_numpy(agg["count"].to_numpy(dtype=np.float32))
        train_pp = pp.transform(agg)
        info["n_train_cells"] = len(agg)
    else:
        train_targets = torch.from_numpy(
            np.clip(pd.to_numeric(train_df[label_col], errors="coerce").fillna(0.0).to_numpy(np.float32), 0.0, 1.0)
        )
        train_weights = torch.ones_like(train_targets)
        train_pp = pp.transform(train_df)
        info["n_train_cells"] = len(train_df)

    val_targets = torch.from_numpy(
        np.clip(pd.to_numeric(val_df[label_col], errors="coerce").fillna(0.0).to_numpy(np.float32), 0.0, 1.0)
    )
    val_weights = torch.ones_like(val_targets)
    val_pp = pp.transform(val_df)

    train_ds = MetadataDataset(train_pp, train_targets, train_weights)
    val_ds = MetadataDataset(val_pp, val_targets, val_weights)
    info["n_train_rows"] = len(train_df)
    info["n_val_rows"] = len(val_df)
    return train_ds, val_ds, info


def _train_one(
    config: LFConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_info_df: pd.DataFrame,
    benchmark_info_df: pd.DataFrame,
    label_col: str,
    device: torch.device,
) -> tuple[LatentFactorModel, MetadataPreprocessor, dict]:
    _seed_everything(config.seed)
    pp = MetadataPreprocessor(model_info_df, benchmark_info_df, config).fit(train_df)
    train_ds, val_ds, ds_info = _build_datasets(train_df, val_df, pp, label_col, config.aggregate)

    print(f"  cells={ds_info['n_train_cells']:,}  val_rows={ds_info['n_val_rows']:,}  "
          f"latent_dim={config.latent_dim}  hidden={config.hidden_dim}  "
          f"layers={config.num_layers}  dropout={config.dropout}  "
          f"wd={config.weight_decay}")

    model = LatentFactorModel(pp, config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    train_info = train_model(model, train_ds, val_ds, config, device)
    final_train_metrics = evaluate(model, train_ds, device, batch_size=config.batch_size)
    final_val_metrics = evaluate(model, val_ds, device, batch_size=config.batch_size)
    out = {
        **ds_info,
        "n_params": n_params,
        "best_val_log_likelihood": train_info["best_val_log_likelihood"],
        "best_epoch": train_info["best_epoch"],
        "wall_seconds": train_info["wall_seconds"],
        "final_train": final_train_metrics,
        "final_val": final_val_metrics,
        "history": train_info["history"],
    }
    return model, pp, out


def _build_submission_folder(
    submission_dir: Path,
    artifacts_dir: Path,
    template_dir: Path,
) -> None:
    submission_dir.mkdir(parents=True, exist_ok=True)
    for name in ("best_model.pt", "preprocessor.pkl", "model_info.csv", "benchmark_info.csv"):
        src = artifacts_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required artifact: {src}")
        shutil.copy2(src, submission_dir / name)
    shutil.copy2(HERE / "latent_factor_pytorch.py", submission_dir / "latent_factor_pytorch.py")
    shutil.copy2(template_dir / "model.py", submission_dir / "model.py")


def _run_official_validation(
    submission_dir: Path,
    val_parquet: Path,
    train_parquet: Path,
    harness_dir: Path,
    out_csv: Path,
    seeds: list[int],
    n: int,
    k: int,
) -> pd.DataFrame:
    cmd = [
        sys.executable,
        str(harness_dir / "scripts" / "run_validation.py"),
        "--submission", str(submission_dir),
        "--val-parquet", str(val_parquet),
        "--train-parquet", str(train_parquet),
        "--N", str(n),
        "--K", str(k),
        "--seeds", *map(str, seeds),
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    proc.check_returncode()
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("seed") or line.startswith("{") or line.startswith("}") or line.startswith("\""):
            continue
        parts = line.split()
        if len(parts) >= 11 and parts[0].isdigit():
            try:
                rows.append(
                    {
                        "seed": int(parts[0]),
                        "n_candidates": int(parts[1]),
                        "n_categories": int(parts[2]),
                        "n_labeled": int(parts[3]),
                        "log_likelihood_excl_labeled": float(parts[-5]),
                        "log_likelihood_incl_all": float(parts[-3]),
                    }
                )
            except (ValueError, IndexError):
                continue
    if not rows:
        rows.append({"seed": -1, "log_likelihood_excl_labeled": float("nan")})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def _baseline_comparison(
    val_df: pd.DataFrame,
    label_col: str,
    final_val_metrics: dict,
    official_seed_lls: list[float],
    logistic_baseline_ll: float,
) -> pd.DataFrame:
    y = np.clip(pd.to_numeric(val_df[label_col], errors="coerce").fillna(0.0).to_numpy(np.float64), 0.0, 1.0)
    yb = (y >= 0.5).astype(int)
    base_rate = float(yb.mean())
    p_const = base_rate
    eps = 1e-6
    ll_const_50 = float(np.mean(yb * np.log(0.5) + (1 - yb) * np.log(0.5)))
    p_safe = np.clip(p_const, eps, 1 - eps)
    ll_const_base = float(np.mean(yb * np.log(p_safe) + (1 - yb) * np.log(1 - p_safe)))

    rows = [
        {"model": "constant_0.5",                "log_likelihood_full_val": ll_const_50,                  "official_mean_ll": ll_const_50},
        {"model": "constant_train_base_rate",    "log_likelihood_full_val": ll_const_base,                "official_mean_ll": float("nan")},
        {"model": "existing_logistic_baseline",  "log_likelihood_full_val": float("nan"),                 "official_mean_ll": logistic_baseline_ll},
        {"model": "latent_factor_pytorch",
         "log_likelihood_full_val": float(final_val_metrics.get("log_likelihood", float("nan"))),
         "official_mean_ll": float(np.mean(official_seed_lls)) if official_seed_lls else float("nan")},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="", type=str,
                    help="starting_kit/Data dir (only needed if splits don't exist).")
    ap.add_argument("--splits-dir", required=True, type=str)
    ap.add_argument("--model-info-csv", required=True, type=str)
    ap.add_argument("--benchmark-info-csv", required=True, type=str)
    ap.add_argument("--validation-harness-dir", required=True, type=str)
    ap.add_argument("--output-dir", required=True, type=str)

    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=0)

    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--id-emb-l2", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--aggregate", action="store_true", default=True)
    ap.add_argument("--no-aggregate", dest="aggregate", action="store_false")

    ap.add_argument("--official-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--official-n", type=int, default=5000)
    ap.add_argument("--official-k", type=int, default=5)

    ap.add_argument("--logistic-baseline-ll", type=float, default=-0.5224,
                    help="Headline LL of the existing logistic baseline (for comparison).")

    ap.add_argument("--sweep", action="store_true", default=False)

    args = ap.parse_args()
    _setup_logging(True)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    _print_env(device)

    print("Loading metadata CSVs ...")
    model_info_df = pd.read_csv(args.model_info_csv)
    benchmark_info_df = pd.read_csv(args.benchmark_info_csv)
    print(f"  model_info: {len(model_info_df)} rows, cols={list(model_info_df.columns)}")
    print(f"  benchmark_info: {len(benchmark_info_df)} rows, cols={list(benchmark_info_df.columns)}")

    train_p, val_p = _ensure_split(args)
    print(f"Loading split: train={train_p}  val={val_p}")
    train_df = pd.read_parquet(train_p)
    val_df = pd.read_parquet(val_p)
    label_col = detect_label_column(train_df)
    print(f"  train rows={len(train_df):,}  val rows={len(val_df):,}  label_col={label_col!r}")

    base_config = LFConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        aggregate=args.aggregate,
        use_amp=args.amp,
        id_emb_l2=args.id_emb_l2,
        seed=args.seed,
    )

    sweeps = [base_config]
    if args.sweep:
        sweeps = []
        for ld in [4, 8, 16, 32]:
            for wd in [1e-4, 1e-3]:
                for dr in [0.05, 0.1, 0.2]:
                    sweeps.append(
                        LFConfig(**{**asdict(base_config),
                                    "latent_dim": ld, "weight_decay": wd, "dropout": dr})
                    )
        print(f"Sweep: {len(sweeps)} runs")

    runs_records: list[dict] = []
    best_run = None

    for run_idx, cfg in enumerate(sweeps, start=1):
        print(f"\n=== Run {run_idx}/{len(sweeps)} ===")
        t0 = time.perf_counter()
        model, pp, info = _train_one(cfg, train_df, val_df, model_info_df, benchmark_info_df, label_col, device)
        rec = {
            "run_idx": run_idx,
            **{k: v for k, v in asdict(cfg).items() if k != "model_categorical" and k != "benchmark_categorical"},
            "n_params": info["n_params"],
            "n_train_cells": info["n_train_cells"],
            "best_epoch": info["best_epoch"],
            "best_val_log_likelihood": info["best_val_log_likelihood"],
            "final_val_log_likelihood": info["final_val"]["log_likelihood"],
            "final_val_brier": info["final_val"]["brier"],
            "final_val_auc": info["final_val"].get("auc_roc", float("nan")),
            "final_train_log_likelihood": info["final_train"]["log_likelihood"],
            "wall_seconds": info["wall_seconds"],
        }
        runs_records.append(rec)
        if (best_run is None) or (rec["final_val_log_likelihood"] > best_run["final_val_log_likelihood"]):
            best_run = rec
            save_artifacts(out_dir, model, pp, cfg, metrics=rec)
            export_metadata_lookup(pp, out_dir)
            best_model_state = (model, pp, cfg, info)
        print(f"  run wall: {time.perf_counter()-t0:.1f}s  final_val_ll={rec['final_val_log_likelihood']:.4f}")

    pd.DataFrame(runs_records).to_csv(out_dir / "runs.csv", index=False)

    print("\n=== Building submission folder + running official-like validation ===")
    submission_dir = out_dir / "submission"
    template_dir = HERE / "submission_template"
    _build_submission_folder(submission_dir, out_dir, template_dir)

    seeds_df = _run_official_validation(
        submission_dir=submission_dir,
        val_parquet=val_p,
        train_parquet=train_p,
        harness_dir=Path(args.validation_harness_dir).resolve(),
        out_csv=out_dir / "official_seeds.csv",
        seeds=list(args.official_seeds),
        n=int(args.official_n),
        k=int(args.official_k),
    )
    seed_lls = seeds_df["log_likelihood_excl_labeled"].dropna().tolist() if "log_likelihood_excl_labeled" in seeds_df.columns else []

    cmp_df = _baseline_comparison(
        val_df=val_df,
        label_col=label_col,
        final_val_metrics={"log_likelihood": best_run["final_val_log_likelihood"]},
        official_seed_lls=seed_lls,
        logistic_baseline_ll=args.logistic_baseline_ll,
    )
    cmp_df.to_csv(out_dir / "baseline_comparison.csv", index=False)

    official_mean = float(np.mean(seed_lls)) if seed_lls else float("nan")
    improvement = official_mean - args.logistic_baseline_ll if seed_lls else float("nan")
    summary = {
        "best_run": best_run,
        "official_mean_log_likelihood": official_mean,
        "improvement_vs_logistic_baseline": improvement,
        "logistic_baseline_ll": args.logistic_baseline_ll,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n" + "=" * 72)
    print(cmp_df.to_string(index=False))
    print("=" * 72)
    print(f"Official-like mean LL across seeds {list(args.official_seeds)}: {official_mean:+.4f}")
    print(f"Logistic baseline reference:        {args.logistic_baseline_ll:+.4f}")
    if not np.isnan(improvement):
        verdict = "BEATS" if improvement > 0 else "TRAILS"
        print(f"Latent-factor model {verdict} logistic by  {improvement:+.4f}")
    train_ll = best_run["final_train_log_likelihood"]
    val_ll = best_run["final_val_log_likelihood"]
    gap = train_ll - val_ll
    print(f"Train/Val gap (overfit indicator): train_ll={train_ll:+.4f}  val_ll={val_ll:+.4f}  gap={gap:+.4f}")
    print(f"All artifacts written under: {out_dir}")


if __name__ == "__main__":
    main()
