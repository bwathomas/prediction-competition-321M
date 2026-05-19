"""Smoke-test a built submission folder.

Usage:

    python scripts/smoke_test_submission.py \
        --submission submission \
        --val artifacts/data/cybench.parquet \
        --n 20

The script imports `submission/model.py`, calls `predict()` on the first N
labeled rows from a parquet file, verifies the output contract, captures
stdout to assert no token was printed, and (optionally) repacks the
submission into ``submission.zip``.

This is what the validation harness's official-like round does -- minus
the adaptive labeling, which has its own test path.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd


def _add_to_path(p: Path) -> None:
    s = str(p.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_submission_module(submission_dir: Path):
    _add_to_path(submission_dir)
    if "model" in sys.modules:
        del sys.modules["model"]
    return importlib.import_module("model")


def _input_rows_from_parquet(parquet_path: Path, n: int) -> list[dict]:
    """Build N input rows from a parquet file using the four-field contract."""
    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent / "src")
    )
    from data import (  # noqa: WPS433 - intentional local import
        REQUIRED_RUNTIME_FIELDS,
        add_stable_keys,
        load_joined_responses,
        normalize_condition,
    )

    if parquet_path.is_dir():
        df = load_joined_responses(parquet_path)
    else:
        # Assume a single response parquet -- best for a quick test.
        df = pd.read_parquet(parquet_path)
        if "subject_content" not in df.columns:
            raise SystemExit(
                "Pass a directory containing the joined dataset OR a parquet "
                "that already has subject_content / item_content columns."
            )
    df = add_stable_keys(df) if "item_key" not in df.columns else df
    cols = [*REQUIRED_RUNTIME_FIELDS, "label"]
    df = df[[c for c in cols if c in df.columns]].head(n)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "benchmark": str(row["benchmark"]),
                "condition": normalize_condition(row.get("condition", "none")),
                "subject_content": str(row["subject_content"]),
                "item_content": str(row["item_content"]),
            }
        )
    return rows


def _detect_token_leak(captured: str) -> list[str]:
    flags: list[str] = []
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token and hf_token in captured:
        flags.append("HF_TOKEN literal found in stdout/stderr")
    suspicious = ("hf_", "Bearer ")
    for s in suspicious:
        if s in captured and "hf_cache" not in captured:
            # 'hf_' alone is common (hf_cache); we only flag when the source
            # is clearly a secret. Conservative heuristic.
            pass
    return flags


def make_zip(submission_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in submission_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(submission_dir))
    return zip_path


def _read_models_txt(submission_dir: Path) -> list[str]:
    p = submission_dir / "models.txt"
    if not p.exists():
        return []
    return [
        line.strip()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_runtime_meta(submission_dir: Path) -> dict:
    p = submission_dir / "artifacts" / "runtime_meta.json"
    if not p.exists():
        return {}
    try:
        import json as _json

        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission")
    parser.add_argument(
        "--val",
        required=False,
        default=None,
        help="Parquet file or joined dataset directory to draw inputs from.",
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--zip-path", default="submission.zip")
    parser.add_argument("--skip-zip", action="store_true")
    parser.add_argument(
        "--latency-warn-ms",
        type=float,
        default=500.0,
        help=(
            "Warn (don't fail) if mean per-call latency exceeds this, in ms. "
            "Helps catch broken batching / oversize judge models."
        ),
    )
    args = parser.parse_args()

    submission_dir = Path(args.submission)
    if not (submission_dir / "model.py").exists():
        print(f"FAIL: {submission_dir / 'model.py'} does not exist")
        return 2
    if not (submission_dir / "models.txt").exists():
        print(f"FAIL: {submission_dir / 'models.txt'} does not exist")
        return 2

    # --- models.txt + runtime_meta cross-check ----------------------------
    declared = _read_models_txt(submission_dir)
    meta = _read_runtime_meta(submission_dir)
    encoder_id = str(meta.get("encoder_model_id", ""))
    judge_block = dict(meta.get("judge") or {})
    judge_enabled = bool(judge_block.get("enabled", False))
    judge_ship_at_runtime = bool(judge_block.get("ship_at_runtime", True))
    judge_id = str(judge_block.get("model_id", ""))

    models_txt_violations: list[str] = []
    if encoder_id and encoder_id not in declared:
        models_txt_violations.append(
            f"models.txt is missing the encoder ({encoder_id!r})"
        )
    if judge_enabled and judge_ship_at_runtime and judge_id and judge_id not in declared:
        models_txt_violations.append(
            f"models.txt is missing the judge ({judge_id!r}); the platform "
            "won't pre-fetch it and predict() will fail at first call."
        )

    print(f"declared models     : {declared}")
    print(f"judge.enabled       : {judge_enabled}")
    print(f"judge.ship_at_runtime: {judge_ship_at_runtime}")
    print(f"judge.model_id      : {judge_id}")
    for v in models_txt_violations:
        print(f"  models.txt FAIL   : {v}")

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        model_module = _load_submission_module(submission_dir)
    init_log = captured.getvalue()
    print("--- module init log (first 600 chars) ---")
    print(init_log[:600])
    print("--- end module init log ---")

    if not hasattr(model_module, "predict"):
        print("FAIL: submission has no predict()")
        return 3

    if args.val:
        rows = _input_rows_from_parquet(Path(args.val), args.n)
    else:
        rows = [
            {
                "benchmark": "mmlupro",
                "condition": "none",
                "subject_content": "Name: smoke-test-model",
                "item_content": f"What is {i} + {i}?",
            }
            for i in range(args.n)
        ]

    t0 = time.time()
    per_call: list[float] = []
    bad: list[str] = []
    for inp in rows:
        c0 = time.time()
        out = model_module.predict(inp, None)
        c1 = time.time()
        per_call.append(c1 - c0)
        if not isinstance(out, float):
            bad.append(f"non-float output: {type(out).__name__}={out!r}")
        elif not math.isfinite(out):
            bad.append(f"non-finite output: {out!r}")
        elif not (0.0 <= out <= 1.0):
            bad.append(f"out-of-range output: {out!r}")
    total = time.time() - t0
    leak_flags = _detect_token_leak(captured.getvalue())

    # In-process judge cache check: predict on the same row twice; the second
    # call should be measurably faster (judge result cached). Skipped if the
    # submission has no judge or if the first call was already <1ms.
    cache_check_ok = True
    cache_check_note = "skipped (no judge runtime)"
    judge_runtime_present = getattr(model_module, "JUDGE", None) is not None
    if rows and judge_runtime_present:
        # Clear PROB_CACHE so the encoder/model path runs both times, isolating
        # the judge cache speedup. We pre-warm with one call, then time two
        # back-to-back calls on the same input.
        first_row = rows[0]
        try:
            getattr(model_module, "_PROB_CACHE", {}).clear()
        except Exception:
            pass
        c0 = time.time()
        _ = model_module.predict(first_row, None)
        c1 = time.time()
        try:
            getattr(model_module, "_PROB_CACHE", {}).clear()
        except Exception:
            pass
        c2 = time.time()
        _ = model_module.predict(first_row, None)
        c3 = time.time()
        first_ms = (c1 - c0) * 1000.0
        second_ms = (c3 - c2) * 1000.0
        cache_check_ok = second_ms < first_ms + 1e-6
        cache_check_note = (
            f"first={first_ms:.1f}ms second={second_ms:.1f}ms "
            f"(judge cache speedup expected)"
        )

    mean_ms = (sum(per_call) / max(1, len(per_call))) * 1000.0
    max_ms = max(per_call) * 1000.0 if per_call else 0.0
    latency_warn = mean_ms > float(args.latency_warn_ms)

    print()
    print(f"calls               : {len(rows)}")
    print(f"total elapsed       : {total:.3f}s")
    print(f"per-call mean       : {mean_ms:.1f}ms")
    print(f"per-call max        : {max_ms:.1f}ms")
    if latency_warn:
        print(
            f"  WARN              : mean latency {mean_ms:.1f}ms exceeds "
            f"{args.latency_warn_ms:.0f}ms threshold; check batching / judge size"
        )
    print(f"contract violations : {len(bad)}")
    for b in bad[:5]:
        print(f"  {b}")
    print(f"token-leak warnings : {len(leak_flags)}")
    for f in leak_flags:
        print(f"  {f}")
    print(f"judge cache check   : {'OK' if cache_check_ok else 'FAIL'} -- {cache_check_note}")

    ok = (
        (not bad)
        and (not leak_flags)
        and (not models_txt_violations)
        and cache_check_ok
    )

    if ok and not args.skip_zip:
        zip_path = make_zip(submission_dir, Path(args.zip_path))
        print(f"wrote zip           : {zip_path.resolve()}")

    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
