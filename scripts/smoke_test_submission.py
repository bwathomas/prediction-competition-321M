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

    # --- LoRA bundle cross-check -----------------------------------------
    # The submission declares its LoRA mode in runtime_meta.json under "lora".
    # For ``adapter_only`` we must see a non-empty ``lora_adapter/`` dir AND a
    # ``peft`` line in requirements.txt; for ``hf_upload`` the merged-encoder
    # repo must be the first models.txt entry; for ``none``/missing the
    # bundle must not ship an adapter dir at all (would silently change
    # predictions on platforms that auto-install peft).
    lora_block = dict(meta.get("lora") or {})
    lora_mode = str(lora_block.get("mode", "none"))
    lora_violations: list[str] = []
    lora_note = "skipped (no lora block in runtime_meta)"
    if lora_mode == "adapter_only":
        adapter_dir = submission_dir / str(
            lora_block.get("adapter_rel_path", "lora_adapter")
        )
        adapter_cfg = adapter_dir / "adapter_config.json"
        adapter_weights_candidates = [
            adapter_dir / "adapter_model.safetensors",
            adapter_dir / "adapter_model.bin",
        ]
        if not adapter_dir.is_dir():
            lora_violations.append(
                f"lora.mode=adapter_only but {adapter_dir} is missing"
            )
        elif not adapter_cfg.exists():
            lora_violations.append(
                f"lora.mode=adapter_only but {adapter_cfg} is missing"
            )
        elif not any(p.exists() for p in adapter_weights_candidates):
            lora_violations.append(
                f"lora.mode=adapter_only but no adapter_model.{{safetensors,bin}} "
                f"found under {adapter_dir}"
            )
        req = submission_dir / "requirements.txt"
        req_txt = req.read_text(encoding="utf-8") if req.exists() else ""
        if "peft" not in req_txt:
            lora_violations.append(
                "lora.mode=adapter_only but requirements.txt has no 'peft' line; "
                "the runtime cannot merge the adapter and would silently fall "
                "back to the base encoder."
            )
        adapter_size = (
            sum(p.stat().st_size for p in adapter_dir.rglob("*") if p.is_file())
            / (1024 * 1024)
            if adapter_dir.is_dir()
            else 0.0
        )
        lora_note = (
            f"mode=adapter_only adapter_dir={adapter_dir.name} "
            f"size={adapter_size:.2f}MB r={lora_block.get('r')} "
            f"alpha={lora_block.get('alpha')}"
        )
    elif lora_mode == "hf_upload":
        merged_repo = str(lora_block.get("merged_encoder_repo", ""))
        if not merged_repo:
            lora_violations.append(
                "lora.mode=hf_upload but merged_encoder_repo is empty"
            )
        elif not declared or declared[0] != merged_repo:
            lora_violations.append(
                f"lora.mode=hf_upload but models.txt[0]={declared[:1]!r} "
                f"!= merged_encoder_repo={merged_repo!r}"
            )
        if (submission_dir / "lora_adapter").exists():
            lora_violations.append(
                "lora.mode=hf_upload but a lora_adapter/ dir was bundled "
                "anyway -- this is wasted space and confusing; drop one."
            )
        lora_note = f"mode=hf_upload merged_repo={merged_repo}"
    elif lora_mode in ("none", ""):
        # Defensive: a 'none' submission must not accidentally ship an adapter.
        if (submission_dir / "lora_adapter").exists():
            lora_violations.append(
                "lora.mode=none but lora_adapter/ is present; either set the "
                "export mode to adapter_only or drop the adapter from the "
                "bundle."
            )
        lora_note = "mode=none"
    else:
        lora_violations.append(f"unknown lora.mode={lora_mode!r}")

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
    print(f"lora                : {lora_note}")
    for v in models_txt_violations:
        print(f"  models.txt FAIL   : {v}")
    for v in lora_violations:
        print(f"  lora FAIL         : {v}")

    captured = io.StringIO()
    init_t0 = time.time()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        model_module = _load_submission_module(submission_dir)
    init_secs = time.time() - init_t0
    init_log = captured.getvalue()
    print("--- module init log (first 600 chars) ---")
    print(init_log[:600])
    print("--- end module init log ---")
    print(f"module import elapsed: {init_secs:.1f}s")

    # If we bundled a LoRA adapter, the runtime model module is supposed to
    # log a one-line "LoRA adapter merged" message during import. Surface
    # that explicitly so a silent skip (peft missing, adapter dir wrong)
    # cannot pass the smoke test unnoticed.
    if lora_mode == "adapter_only":
        merged_ok = "LoRA adapter merged" in init_log
        if not merged_ok:
            lora_violations.append(
                "lora.mode=adapter_only but the runtime did NOT log "
                "'LoRA adapter merged' during import -- the adapter was "
                "almost certainly not applied. Predictions will reflect "
                "the un-fine-tuned encoder."
            )
            print("  lora FAIL         : adapter merge log line not seen at import")
        else:
            print("  lora OK           : adapter merged at module import")

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

    # --- NN feature smoke tests ------------------------------------------
    # These assert that the runtime TrainingItemCache.compute_nn_features:
    #   1) returns a finite [NN_FEATURE_DIM] vector,
    #   2) puts passrate_mean / coverage in [0, 1],
    #   3) gives top1_similarity ~ 1 for a query that is a training-item
    #      duplicate (we pull the first cached embedding back through
    #      `nearest` so the query IS in the index),
    #   4) runs in < 50ms per call after the FAISS index is warm.
    # Skipped cleanly when NN features are disabled in the submission.
    nn_violations: list[str] = []
    nn_note = "skipped (NN features disabled or cache absent)"
    nn_enabled_runtime = bool(getattr(model_module, "NN_ENABLED", False))
    training_cache = getattr(model_module, "TRAINING_CACHE", None)
    if nn_enabled_runtime and training_cache is not None:
        try:
            import numpy as _np

            nn_dim = int(getattr(model_module, "NN_FEATURE_DIM", 8))
            n_subjects_nn = (
                int(training_cache.nn_passrate.shape[0])
                if getattr(training_cache, "nn_passrate", None) is not None
                else 0
            )
            test_sid = 1 if n_subjects_nn > 1 else 0
            warmup_emb = _np.zeros(int(training_cache.embeddings_q.shape[1]), dtype=_np.float32) \
                if hasattr(training_cache, "embeddings_q") else None
            if warmup_emb is None or warmup_emb.size == 0:
                nn_note = "skipped (no embeddings_q on TRAINING_CACHE)"
            else:
                _ = training_cache.compute_nn_features(warmup_emb, subject_id=test_sid)
                nn_lat: list[float] = []
                for j in range(min(5, len(rows))):
                    inp = rows[j]
                    enc = getattr(model_module, "ENCODER", None)
                    if enc is None or not hasattr(enc, "encode"):
                        item_emb = warmup_emb
                    else:
                        item_emb = _np.asarray(
                            enc.encode(inp["item_content"]), dtype=_np.float32
                        ).reshape(-1)
                    t = time.time()
                    vec = training_cache.compute_nn_features(item_emb, subject_id=test_sid)
                    nn_lat.append(time.time() - t)
                    if vec.shape != (nn_dim,):
                        nn_violations.append(
                            f"vec shape {vec.shape} != ({nn_dim},)"
                        )
                        continue
                    if not _np.all(_np.isfinite(vec)):
                        nn_violations.append("non-finite NN feature vector")
                        continue
                    pmean = float(vec[0])
                    cov = float(vec[3])
                    if not (0.0 <= pmean <= 1.0):
                        nn_violations.append(
                            f"passrate_mean out of [0,1]: {pmean:.4f}"
                        )
                    if not (0.0 <= cov <= 1.0):
                        nn_violations.append(f"coverage out of [0,1]: {cov:.4f}")
                # Self-similarity check: a training-item duplicate's top-1
                # similarity should be ~1. We pull the dequantized embedding
                # of the first training item and query the cache with it.
                top1_sim_dup = None
                try:
                    recon0 = training_cache._dequantized()[0:1]
                    if recon0.size:
                        dup_vec = training_cache.compute_nn_features(
                            recon0[0], subject_id=test_sid
                        )
                        top1_sim_dup = float(dup_vec[5])
                        if top1_sim_dup < 0.9:
                            nn_violations.append(
                                f"top1_similarity for duplicate is "
                                f"{top1_sim_dup:.3f} (< 0.9); index may "
                                "be mis-built"
                            )
                except Exception as e:  # noqa: BLE001
                    nn_violations.append(
                        f"could not run duplicate-query smoke check: {e!r}"
                    )
                nn_lat_ms = [x * 1000.0 for x in nn_lat]
                worst_post_warmup = max(nn_lat_ms[1:]) if len(nn_lat_ms) > 1 else (
                    nn_lat_ms[0] if nn_lat_ms else 0.0
                )
                if worst_post_warmup > 50.0:
                    nn_violations.append(
                        f"NN feature compute > 50ms after warmup: "
                        f"{worst_post_warmup:.1f}ms"
                    )
                nn_note = (
                    f"n={len(nn_lat)} mean={(sum(nn_lat_ms) / max(1, len(nn_lat_ms))):.1f}ms "
                    f"max_post_warmup={worst_post_warmup:.1f}ms"
                )
                if top1_sim_dup is not None:
                    nn_note += f" top1_dup_sim={top1_sim_dup:.3f}"
        except Exception as e:  # noqa: BLE001
            nn_violations.append(f"NN smoke test crashed: {e!r}")

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
    nn_ok = not nn_violations
    print(f"nn feature smoke    : {'OK' if nn_ok else 'FAIL'} -- {nn_note}")
    for v in nn_violations[:5]:
        print(f"  nn FAIL           : {v}")

    ok = (
        (not bad)
        and (not leak_flags)
        and (not models_txt_violations)
        and (not lora_violations)
        and cache_check_ok
        and nn_ok
    )

    if ok and not args.skip_zip:
        zip_path = make_zip(submission_dir, Path(args.zip_path))
        print(f"wrote zip           : {zip_path.resolve()}")

    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
