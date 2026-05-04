"""End-to-end runner for the metadata-only latent-factor model.

This is the single command Colab calls. It:

1. Loads the joined responses parquet (from prepare_split.py output).
2. Fits the metadata preprocessor on TRAIN ONLY  *(once)*.
3. Aggregates train rows into (model, benchmark, condition) cells *(once)*.
4. Caches everything to disk so parallel sweep workers reload instantly.
5. Runs one OR many configs through the trainer; if `--parallel-runs > 1`
   AND CUDA is available, runs them concurrently on the same GPU via
   ProcessPoolExecutor (spawn) so the A100 is kept busy.
6. Builds a submission folder around the best config and runs the
   official-like validation harness across 3 seeds.
7. Writes:
     outputs/latent_factor/best_model.pt
     outputs/latent_factor/preprocessor.pkl
     outputs/latent_factor/metrics.json
     outputs/latent_factor/runs.csv
     outputs/latent_factor/baseline_comparison.csv
     outputs/latent_factor/runs/run_XYZ/...   (per-run artifacts)

Sweep dimensions (extended): latent_dim, hidden_dim, num_layers, dropout,
weight_decay, lr, id_emb_l2, patience.

Example (Colab, A100):

    python run_latent_factor_colab.py \\
        --data-dir starting_kit/Data \\
        --splits-dir validation_harness/splits/v1 \\
        --model-info-csv starting_kit/Model_Info/model_info.csv \\
        --benchmark-info-csv starting_kit/benchmark_info/benchmark_info.csv \\
        --validation-harness-dir validation_harness \\
        --output-dir outputs/latent_factor \\
        --sweep --sweep-budget 24 --parallel-runs 8 --amp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import pickle
import random
import shutil
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# Logging / setup helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = True, run_label: str = "") -> None:
    """Reset root logging with the (possibly run-tagged) format.

    Called from both the parent process and each subprocess worker.
    """
    level = logging.INFO if verbose else logging.WARNING
    fmt = "%(asctime)s " + (f"[{run_label}] " if run_label else "") + "%(message)s"
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", force=True)


def _seed_everything(seed: int) -> None:
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


def _file_md5(path: Path, chunk: int = 4 * 1024 * 1024) -> str | None:
    """Best-effort MD5 of a file. Returns None on failure / missing files."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def _capture_git(repo_root: Path) -> dict[str, Any]:
    """Capture git commit + dirty state for reproducibility (best-effort)."""
    out: dict[str, Any] = {"repo_root": str(repo_root)}
    try:
        out["commit"] = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        out["commit"] = None
    try:
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True,
        )
        out["dirty"] = bool(status.strip())
    except Exception:
        out["dirty"] = None
    try:
        out["branch"] = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        out["branch"] = None
    return out


def _capture_packages() -> dict[str, str]:
    """Snapshot versions of the libraries this script depends on."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("torch", "numpy", "pandas", "pyarrow", "sklearn"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "?")
        except ImportError:
            versions[name] = "<not installed>"
    return versions


def _save_reproduce_files(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    base_config: LFConfig,
    best_run: dict,
    train_parquet: Path,
    val_parquet: Path,
    repo_root: Path,
    sweep_seconds: float,
    n_runs: int,
    parallel_runs: int,
    summary: dict,
) -> None:
    """Write the JSON manifest + shell script needed to reproduce best_model.pt.

    The manifest is intentionally exhaustive: env, git commit, package
    versions, the FULL launch argv, the best run's config (post-sweep), the
    seeds, and md5s of the train/val parquets so a re-run can prove it had
    the same data.
    """
    repro: dict[str, Any] = {
        "command_argv": [sys.executable, sys.argv[0], *sys.argv[1:]],
        "args_namespace": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "best_run_idx": best_run["run_idx"],
        "best_config": {k: best_run[k] for k in (
            "latent_dim", "hidden_dim", "num_layers", "dropout",
            "lr", "weight_decay", "id_emb_l2", "patience",
            "batch_size", "epochs", "aggregate", "use_amp", "seed",
            "max_embedding_dim", "min_token_count", "grad_clip",
        ) if k in best_run},
        "base_config": asdict(base_config),
        "seeds": {
            "model_seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "official_seeds": list(args.official_seeds),
        },
        "data": {
            "train_parquet": str(train_parquet),
            "val_parquet": str(val_parquet),
            "train_md5": _file_md5(train_parquet),
            "val_md5": _file_md5(val_parquet),
            "model_info_csv": str(args.model_info_csv),
            "benchmark_info_csv": str(args.benchmark_info_csv),
            "model_info_md5": _file_md5(Path(args.model_info_csv)),
            "benchmark_info_md5": _file_md5(Path(args.benchmark_info_csv)),
        },
        "git": _capture_git(repo_root),
        "packages": _capture_packages(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sweep_seconds": sweep_seconds,
        "n_runs": n_runs,
        "parallel_runs": parallel_runs,
        "summary": summary,
        "artifacts": {
            "best_model_pt": "best_model.pt",
            "weights_pt": "weights.pt",
            "preprocessor_pkl": "preprocessor.pkl",
            "model_info_csv": "model_info.csv",
            "benchmark_info_csv": "benchmark_info.csv",
            "submission_dir": "submission/",
            "per_run_dir": f"runs/run_{best_run['run_idx']:03d}/",
        },
    }
    (out_dir / "reproduce.json").write_text(json.dumps(repro, indent=2, default=float))

    # One-line CLI to retrain the best config (forces single run, no sweep).
    best_args = [
        "--data-dir",                str(args.data_dir),
        "--splits-dir",              str(args.splits_dir),
        "--model-info-csv",          str(args.model_info_csv),
        "--benchmark-info-csv",      str(args.benchmark_info_csv),
        "--validation-harness-dir",  str(args.validation_harness_dir),
        "--output-dir",              str(out_dir / "reproduced"),
        "--latent-dim",  str(repro["best_config"].get("latent_dim", base_config.latent_dim)),
        "--hidden-dim",  str(repro["best_config"].get("hidden_dim", base_config.hidden_dim)),
        "--num-layers",  str(repro["best_config"].get("num_layers", base_config.num_layers)),
        "--dropout",     str(repro["best_config"].get("dropout", base_config.dropout)),
        "--weight-decay", str(repro["best_config"].get("weight_decay", base_config.weight_decay)),
        "--id-emb-l2",   str(repro["best_config"].get("id_emb_l2", base_config.id_emb_l2)),
        "--lr",          str(repro["best_config"].get("lr", base_config.lr)),
        "--batch-size",  str(repro["best_config"].get("batch_size", base_config.batch_size)),
        "--epochs",      str(repro["best_config"].get("epochs", base_config.epochs)),
        "--patience",    str(repro["best_config"].get("patience", base_config.patience)),
        "--seed",        str(repro["best_config"].get("seed", base_config.seed)),
        "--split-seed",  str(args.split_seed),
        "--val-fraction", str(args.val_fraction),
        "--official-seeds", *map(str, args.official_seeds),
        "--official-n", str(args.official_n),
        "--official-k", str(args.official_k),
        "--logistic-baseline-ll", str(args.logistic_baseline_ll),
    ]
    if args.amp:
        best_args.append("--amp")
    else:
        best_args.append("--no-amp")
    if args.aggregate:
        best_args.append("--aggregate")
    else:
        best_args.append("--no-aggregate")

    py = sys.executable
    script = sys.argv[0]
    sh = "#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(
        [shlex.quote(py), shlex.quote(script), *(shlex.quote(a) for a in best_args)]
    ) + "\n"
    bat = "@echo off\n\"" + py + "\" \"" + script + "\" " + " ".join(
        ('"' + a + '"' if " " in a else a) for a in best_args
    ) + "\n"
    (out_dir / "reproduce.sh").write_text(sh)
    try:
        os.chmod(out_dir / "reproduce.sh", 0o755)
    except OSError:
        pass
    (out_dir / "reproduce.bat").write_text(bat)


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


# ---------------------------------------------------------------------------
# Precompute + cache shared artifacts (preprocessor + transformed tensors)
# ---------------------------------------------------------------------------


def _precompute_and_cache(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_info_df: pd.DataFrame,
    benchmark_info_df: pd.DataFrame,
    label_col: str,
    base_config: LFConfig,
    cache_dir: Path,
) -> dict:
    """Fit preprocessor + transform train/val tensors ONCE and pickle them.

    Returns a small "manifest" dict that workers use to reload everything.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    pp = MetadataPreprocessor(model_info_df, benchmark_info_df, base_config).fit(train_df)

    if base_config.aggregate:
        agg = aggregate_cells(train_df, label_col=label_col)
        train_pp = pp.transform(agg)
        train_targets = torch.from_numpy(agg["target"].to_numpy(dtype=np.float32))
        train_weights = torch.from_numpy(agg["count"].to_numpy(dtype=np.float32))
        n_train_cells = len(agg)
    else:
        train_pp = pp.transform(train_df)
        train_targets = torch.from_numpy(
            np.clip(
                pd.to_numeric(train_df[label_col], errors="coerce").fillna(0.0).to_numpy(np.float32),
                0.0, 1.0,
            )
        )
        train_weights = torch.ones_like(train_targets)
        n_train_cells = len(train_df)

    val_pp = pp.transform(val_df)
    val_targets = torch.from_numpy(
        np.clip(
            pd.to_numeric(val_df[label_col], errors="coerce").fillna(0.0).to_numpy(np.float32),
            0.0, 1.0,
        )
    )
    val_weights = torch.ones_like(val_targets)

    with open(cache_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(pp, f)
    torch.save(train_pp, cache_dir / "train_pp.pt")
    torch.save(val_pp, cache_dir / "val_pp.pt")
    torch.save(train_targets, cache_dir / "train_targets.pt")
    torch.save(train_weights, cache_dir / "train_weights.pt")
    torch.save(val_targets, cache_dir / "val_targets.pt")
    torch.save(val_weights, cache_dir / "val_weights.pt")

    manifest = {
        "cache_dir": str(cache_dir.resolve()),
        "n_train_rows": int(len(train_df)),
        "n_train_cells": int(n_train_cells),
        "n_val_rows": int(len(val_df)),
        "label_col": label_col,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _load_cache(cache_dir: Path):
    """Load everything precomputed into the worker process."""
    with open(cache_dir / "preprocessor.pkl", "rb") as f:
        pp = pickle.load(f)
    train_pp = torch.load(cache_dir / "train_pp.pt", weights_only=False)
    val_pp = torch.load(cache_dir / "val_pp.pt", weights_only=False)
    train_targets = torch.load(cache_dir / "train_targets.pt", weights_only=False)
    train_weights = torch.load(cache_dir / "train_weights.pt", weights_only=False)
    val_targets = torch.load(cache_dir / "val_targets.pt", weights_only=False)
    val_weights = torch.load(cache_dir / "val_weights.pt", weights_only=False)
    return pp, (train_pp, train_targets, train_weights), (val_pp, val_targets, val_weights)


# ---------------------------------------------------------------------------
# Worker (module-level so it pickles for spawn)
# ---------------------------------------------------------------------------


def _worker_train(args_dict: dict) -> dict:
    """Train one config. Safe for ProcessPoolExecutor + spawn.

    The worker re-imports `latent_factor_pytorch` (since spawn doesn't
    inherit anything) and reloads precomputed tensors from disk.
    """
    run_idx = int(args_dict["run_idx"])
    total = int(args_dict["total"])
    label = f"run {run_idx:03d}/{total:03d}"

    _setup_logging(True, run_label=label)
    log = logging.getLogger("latent_factor")

    cache_dir = Path(args_dict["cache_dir"])
    runs_dir = Path(args_dict["runs_dir"])
    config_dict = dict(args_dict["config"])
    # Tuples are not directly representable in JSON / asdict serialization
    # safety; restore them.
    if "model_categorical" in config_dict and isinstance(config_dict["model_categorical"], list):
        config_dict["model_categorical"] = tuple(config_dict["model_categorical"])
    if "benchmark_categorical" in config_dict and isinstance(config_dict["benchmark_categorical"], list):
        config_dict["benchmark_categorical"] = tuple(config_dict["benchmark_categorical"])
    config = LFConfig(**config_dict)

    _seed_everything(config.seed)

    pp, (train_pp, train_targets, train_weights), (val_pp, val_targets, val_weights) = _load_cache(cache_dir)
    train_ds = MetadataDataset(train_pp, train_targets, train_weights)
    val_ds = MetadataDataset(val_pp, val_targets, val_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LatentFactorModel(pp, config)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "config: latent_dim=%d hidden=%d layers=%d dropout=%.2f wd=%.1e lr=%.1e "
        "id_l2=%.1e patience=%d  params=%d",
        config.latent_dim, config.hidden_dim, config.num_layers, config.dropout,
        config.weight_decay, config.lr, config.id_emb_l2, config.patience, n_params,
    )

    t0 = time.perf_counter()
    train_info = train_model(model, train_ds, val_ds, config, device, run_label=label)
    final_train = evaluate(model, train_ds, device, batch_size=config.batch_size)
    final_val = evaluate(model, val_ds, device, batch_size=config.batch_size)
    wall = time.perf_counter() - t0

    rec = {
        "run_idx": run_idx,
        "n_params": int(n_params),
        "best_val_log_likelihood": float(train_info["best_val_log_likelihood"]),
        "best_epoch": int(train_info["best_epoch"]),
        "final_val_log_likelihood": float(final_val["log_likelihood"]),
        "final_val_brier": float(final_val["brier"]),
        "final_val_auc": float(final_val.get("auc_roc", float("nan"))),
        "final_train_log_likelihood": float(final_train["log_likelihood"]),
        "wall_seconds": float(wall),
    }
    for k, v in asdict(config).items():
        if k in ("model_categorical", "benchmark_categorical"):
            continue
        rec[k] = v

    run_dir = runs_dir / f"run_{run_idx:03d}"
    save_artifacts(run_dir, model, pp, config, metrics=rec)
    export_metadata_lookup(pp, run_dir)

    log.info(
        "DONE run=%d  best_val_ll=%+.4f@%d  final_val_ll=%+.4f  "
        "final_train_ll=%+.4f  wall=%.1fs",
        run_idx, rec["best_val_log_likelihood"], rec["best_epoch"],
        rec["final_val_log_likelihood"], rec["final_train_log_likelihood"], wall,
    )
    return rec


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------


def _build_full_grid(base: LFConfig) -> list[LFConfig]:
    """Cartesian product over the seven sweep dimensions."""
    grid = []
    for ld in (4, 8, 16, 32):
        for hd in (128, 256):
            for dr in (0.05, 0.1, 0.2):
                for wd in (1e-4, 1e-3):
                    for lr in (1e-3, 3e-3):
                        for id_l2 in (1e-4, 1e-3):
                            for patience in (5, 10):
                                grid.append(
                                    LFConfig(
                                        **{
                                            **asdict(base),
                                            "latent_dim": ld,
                                            "hidden_dim": hd,
                                            "dropout": dr,
                                            "weight_decay": wd,
                                            "lr": lr,
                                            "id_emb_l2": id_l2,
                                            "patience": patience,
                                        }
                                    )
                                )
    return grid


def _build_sweep(base: LFConfig, mode: str, budget: int, seed: int) -> list[LFConfig]:
    """Return either the full grid (mode='full') or a random subset."""
    full = _build_full_grid(base)
    if mode == "full" or budget <= 0 or budget >= len(full):
        return full
    rng = random.Random(seed)
    return rng.sample(full, budget)


# ---------------------------------------------------------------------------
# Submission folder + official validation
# ---------------------------------------------------------------------------


def _build_submission_folder(submission_dir: Path, artifacts_dir: Path, template_dir: Path) -> None:
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
    final_val_ll: float,
    official_seed_lls: list[float],
    logistic_baseline_ll: float,
) -> pd.DataFrame:
    y = np.clip(pd.to_numeric(val_df[label_col], errors="coerce").fillna(0.0).to_numpy(np.float64), 0.0, 1.0)
    yb = (y >= 0.5).astype(int)
    base_rate = float(yb.mean())
    eps = 1e-6
    ll_const_50 = float(np.mean(yb * np.log(0.5) + (1 - yb) * np.log(0.5)))
    p_safe = float(np.clip(base_rate, eps, 1 - eps))
    ll_const_base = float(np.mean(yb * np.log(p_safe) + (1 - yb) * np.log(1 - p_safe)))

    rows = [
        {"model": "constant_0.5",                "log_likelihood_full_val": ll_const_50,                  "official_mean_ll": ll_const_50},
        {"model": "constant_train_base_rate",    "log_likelihood_full_val": ll_const_base,                "official_mean_ll": float("nan")},
        {"model": "existing_logistic_baseline",  "log_likelihood_full_val": float("nan"),                 "official_mean_ll": logistic_baseline_ll},
        {"model": "latent_factor_pytorch",
         "log_likelihood_full_val": float(final_val_ll),
         "official_mean_ll": float(np.mean(official_seed_lls)) if official_seed_lls else float("nan")},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _default_parallel_runs() -> int:
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info(0)
            est = max(2, min(8, int(free / (3 * 10**9))))
            return est
        except Exception:
            return 4
    return max(1, (os.cpu_count() or 2) // 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="", type=str)
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

    ap.add_argument(
        "--log-every-steps", type=int, default=10,
        help="Print a one-line training-step summary every N optimizer steps. "
             "Set to 0 to disable per-step prints (per-epoch summary still emitted).",
    )
    ap.add_argument(
        "--progress-bar", dest="progress_bar", action="store_true", default=True,
        help="Show a tqdm bar over total training steps with live ETA. (default on)",
    )
    ap.add_argument(
        "--no-progress-bar", dest="progress_bar", action="store_false",
        help="Disable the tqdm progress bar (e.g. for non-interactive logs).",
    )

    ap.add_argument("--official-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--official-n", type=int, default=5000)
    ap.add_argument("--official-k", type=int, default=5)

    ap.add_argument("--logistic-baseline-ll", type=float, default=-0.5224)

    ap.add_argument("--sweep", action="store_true", default=False)
    ap.add_argument("--sweep-mode", choices=("random", "full"), default="random")
    ap.add_argument("--sweep-budget", type=int, default=24,
                    help="Random subset size when --sweep-mode=random (default 24).")
    ap.add_argument("--parallel-runs", type=int, default=0,
                    help="0 = auto-pick. >1 trains multiple configs in parallel "
                         "via ProcessPoolExecutor + spawn.")
    ap.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Path to a previous --output-dir (or per-run dir) containing "
             "best_model.pt + preprocessor.pkl. Skips training, copies the "
             "checkpoint into the new output dir, builds the submission folder "
             "and runs official-like validation only.",
    )

    args = ap.parse_args()
    _setup_logging(True)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    _print_env(device)

    print("Loading metadata CSVs ...")
    model_info_df = pd.read_csv(args.model_info_csv)
    benchmark_info_df = pd.read_csv(args.benchmark_info_csv)
    print(f"  model_info: {len(model_info_df)} rows, cols={list(model_info_df.columns)}")
    print(f"  benchmark_info: {len(benchmark_info_df)} rows, cols={list(benchmark_info_df.columns)}")

    if args.resume_from:
        resume_dir = Path(args.resume_from).resolve()
        print(f"\n--resume-from supplied: skipping training, loading checkpoint from {resume_dir}")
        for fname in ("best_model.pt", "preprocessor.pkl", "model_info.csv", "benchmark_info.csv"):
            src = resume_dir / fname
            if not src.exists():
                raise SystemExit(f"Required file missing in --resume-from: {src}")
            shutil.copy2(src, out_dir / fname)
        if (resume_dir / "weights.pt").exists():
            shutil.copy2(resume_dir / "weights.pt", out_dir / "weights.pt")

        train_p, val_p = _ensure_split(args)
        print(f"Loading split: train={train_p}  val={val_p}")
        val_df = pd.read_parquet(val_p)
        label_col = detect_label_column(val_df)

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
        seed_lls = (
            seeds_df["log_likelihood_excl_labeled"].dropna().tolist()
            if "log_likelihood_excl_labeled" in seeds_df.columns else []
        )
        official_mean = float(np.mean(seed_lls)) if seed_lls else float("nan")
        improvement = official_mean - args.logistic_baseline_ll if seed_lls else float("nan")
        verdict = "BEATS" if (not np.isnan(improvement) and improvement > 0) else "TRAILS"
        print("\n" + "=" * 72)
        print(f"Official-like mean LL across seeds {list(args.official_seeds)}: {official_mean:+.4f}")
        print(f"Logistic baseline reference:        {args.logistic_baseline_ll:+.4f}")
        if not np.isnan(improvement):
            print(f"Latent-factor model {verdict} logistic by  {improvement:+.4f}")
        print(f"All artifacts written under: {out_dir}")
        return

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
        log_every_steps=int(args.log_every_steps),
        show_progress_bar=bool(args.progress_bar),
    )

    print(f"Precomputing preprocessor + tensors -> {cache_dir} ...")
    manifest = _precompute_and_cache(
        train_df=train_df, val_df=val_df,
        model_info_df=model_info_df, benchmark_info_df=benchmark_info_df,
        label_col=label_col, base_config=base_config,
        cache_dir=cache_dir,
    )
    print(f"  cells={manifest['n_train_cells']:,}  val_rows={manifest['n_val_rows']:,}  "
          f"  precompute_seconds={manifest['elapsed_seconds']:.1f}")

    if args.sweep:
        sweeps = _build_sweep(base_config, args.sweep_mode, args.sweep_budget, args.seed)
    else:
        sweeps = [base_config]

    parallel = args.parallel_runs if args.parallel_runs > 0 else _default_parallel_runs()
    parallel = max(1, min(parallel, len(sweeps)))

    # When more than one worker runs concurrently, multiple tqdm bars from
    # different processes interleave very badly in stdout. Disable the bar
    # for parallel sweeps; per-step + per-epoch text logs still go through.
    if parallel > 1:
        from dataclasses import replace
        sweeps = [replace(c, show_progress_bar=False) for c in sweeps]
        print(
            f"Sweep: {len(sweeps)} runs ({args.sweep_mode if args.sweep else 'single'}), "
            f"parallel_runs={parallel}, device={device}  "
            f"(progress bar disabled in parallel mode; use --parallel-runs 1 for the bar)"
        )
    else:
        print(
            f"Sweep: {len(sweeps)} runs ({args.sweep_mode if args.sweep else 'single'}), "
            f"parallel_runs={parallel}, device={device}  "
            f"(sequential -> per-run tqdm bar is shown)"
        )

    runs_records: list[dict] = []
    sweep_t0 = time.perf_counter()

    def _make_args(run_idx: int, cfg: LFConfig) -> dict:
        return {
            "run_idx": run_idx,
            "total": len(sweeps),
            "cache_dir": str(cache_dir),
            "runs_dir": str(runs_dir),
            "config": asdict(cfg),
        }

    if parallel == 1:
        for run_idx, cfg in enumerate(sweeps, start=1):
            print(f"\n=== Sweep {run_idx}/{len(sweeps)} (sequential) ===")
            rec = _worker_train(_make_args(run_idx, cfg))
            runs_records.append(rec)
            done = len(runs_records)
            elapsed = time.perf_counter() - sweep_t0
            avg = elapsed / done
            eta_s = avg * (len(sweeps) - done)
            best_so_far = max(r["final_val_log_likelihood"] for r in runs_records)
            print(f"  >>> sweep progress: {done}/{len(sweeps)} done  elapsed={elapsed:.1f}s  "
                  f"eta~{eta_s:.1f}s  best_val_ll={best_so_far:+.4f}")
    else:
        try:
            mp_ctx = torch.multiprocessing.get_context("spawn")
        except RuntimeError:
            import multiprocessing as _mp
            mp_ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=parallel, mp_context=mp_ctx) as executor:
            futures = {}
            for run_idx, cfg in enumerate(sweeps, start=1):
                fut = executor.submit(_worker_train, _make_args(run_idx, cfg))
                futures[fut] = run_idx
            for fut in as_completed(futures):
                run_idx = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    print(f"!!! run {run_idx:03d} failed: {type(e).__name__}: {e}")
                    continue
                runs_records.append(rec)
                done = len(runs_records)
                elapsed = time.perf_counter() - sweep_t0
                avg = elapsed / done
                rem = len(sweeps) - done
                eta_s = avg * rem / parallel
                best_so_far = max(r["final_val_log_likelihood"] for r in runs_records)
                print(
                    f"  >>> sweep progress: {done}/{len(sweeps)} done  "
                    f"elapsed={elapsed:.1f}s  eta~{eta_s:.1f}s  "
                    f"best_val_ll={best_so_far:+.4f}",
                    flush=True,
                )

    if not runs_records:
        raise SystemExit("All sweep runs failed.")

    pd.DataFrame(runs_records).sort_values("final_val_log_likelihood", ascending=False).to_csv(
        out_dir / "runs.csv", index=False
    )

    best_run = max(runs_records, key=lambda r: r["final_val_log_likelihood"])
    best_dir = runs_dir / f"run_{best_run['run_idx']:03d}"
    print(f"\nBest run: {best_run['run_idx']} (final_val_ll={best_run['final_val_log_likelihood']:+.4f})")

    for fname in ("best_model.pt", "weights.pt", "preprocessor.pkl",
                  "model_info.csv", "benchmark_info.csv", "metrics.json"):
        src = best_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

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
    seed_lls = (
        seeds_df["log_likelihood_excl_labeled"].dropna().tolist()
        if "log_likelihood_excl_labeled" in seeds_df.columns else []
    )

    cmp_df = _baseline_comparison(
        val_df=val_df,
        label_col=label_col,
        final_val_ll=best_run["final_val_log_likelihood"],
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
        "sweep_seconds": time.perf_counter() - sweep_t0,
        "n_runs": len(runs_records),
        "parallel_runs": parallel,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    _save_reproduce_files(
        out_dir,
        args=args,
        base_config=base_config,
        best_run=best_run,
        train_parquet=train_p,
        val_parquet=val_p,
        repo_root=HERE.parent,
        sweep_seconds=summary["sweep_seconds"],
        n_runs=summary["n_runs"],
        parallel_runs=summary["parallel_runs"],
        summary={
            "official_mean_log_likelihood": official_mean,
            "improvement_vs_logistic_baseline": improvement,
            "logistic_baseline_ll": args.logistic_baseline_ll,
            "best_run_final_val_ll": best_run["final_val_log_likelihood"],
            "best_run_final_train_ll": best_run["final_train_log_likelihood"],
            "best_run_best_val_ll": best_run["best_val_log_likelihood"],
            "best_run_best_epoch": best_run["best_epoch"],
        },
    )

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
    print(f"Sweep wall: {summary['sweep_seconds']:.1f}s ({len(runs_records)} runs, "
          f"{parallel} in parallel)")
    print(f"All artifacts written under: {out_dir}")
    print(
        "Reproducibility: weights.pt + best_model.pt + preprocessor.pkl saved; "
        "see reproduce.json (full manifest + data hashes + git + env) and "
        "reproduce.sh / reproduce.bat (one-line CLI to retrain just the best config)."
    )


if __name__ == "__main__":
    if torch.cuda.is_available():
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    main()
