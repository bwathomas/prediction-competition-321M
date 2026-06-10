"""3-family ensemble: Qwen3-Embedding-8B + llama-embed-nemotron-8b + LGAI-Embedding-Preview.

Architecture (mirrors the proven ensemble_3way_logit_fwls orchestrator, with one more
layer of depth and fitted linear weights instead of a plain logit mean):

  L1  per family: linear logit blend over the PRUNED mlp leave-one-group-out members
  L2  per family: linear logit blend of [mlp_L1, etbig numpy forest]
  L3  across families: linear logit blend of the 3 family probabilities (weights+bias
      fitted offline on honest GroupKFold(item) OOF; see artifacts/stack_top.json)
  CAL on top: the trc5 type-conditional partial-pool intercept calibrator, refit within
      each round on the `labeled` rows (per-benchmark intercepts shrunk toward
      b_global + delta_type*is_new_bc; ridge 20/20/10; Newton).

Unlike ensemble_3way (three divergent third-party sub-bundles loaded via importlib), all
three families here share one code path (fam_common.py / encoders.py) — family differences
live entirely in <fam>/artifacts/runtime_meta.json. Robustness contract is the same: a
family that fails predict is dropped from the blend; all three failing returns DEFAULT_PROB.

Streamed-flush (trc5 streamed_flush_v1): labeling.py calls _enqueue_for_batch once per
acquisition candidate; each call flushes at most one encoder batch per family, so by the
first predict() nearly all item embeddings are cached. predict() drains the small residue.
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fam_common as fc                                            # noqa: E402
from encoders import Encoder                                       # noqa: E402

LOG = logging.getLogger("ensemble3v2")
logging.basicConfig(level=logging.INFO)

DEFAULT_PROB = 0.5
EPS = 1e-6
_LOGIT_CLAMP = 20.0

normalize_condition = fc.normalize_condition
stable_sha256 = fc.stable_sha256

# ---------------------------------------------------------------------------------
# shared artifacts
# ---------------------------------------------------------------------------------
_SHARED_DIR = HERE / "shared_artifacts"
_PASSRATE = fc.CsrPassrate.load(_SHARED_DIR / "passrate.npz")
_SUBJ_VOCAB = {str(k): int(v) for k, v in json.loads(
    (_SHARED_DIR / "subj_vocab.json").read_text(encoding="utf-8")).items()}
_PASSRATE_ROW = {str(k): int(v) for k, v in json.loads(
    (_SHARED_DIR / "passrate_row.json").read_text(encoding="utf-8")).items()}
_BC_TO_ID = {str(k): int(v) for k, v in json.loads(
    (_SHARED_DIR / "bc_to_id.json").read_text(encoding="utf-8")).items()}
_train_counts = json.loads((_SHARED_DIR / "train_counts.json").read_text(encoding="utf-8"))
_N_TRAIN_PER_BC = {str(k): int(v) for k, v in _train_counts["n_per_bc"].items()}
_N_TRAIN_PER_SUBJECT = {str(k): int(v)
                        for k, v in _train_counts["n_per_subject"].items()}
_SUBJECT_TO_ID = dict(_SUBJ_VOCAB)   # labeling.py export (subject_key -> id)
_STACK_TOP = json.loads((_SHARED_DIR / "stack_top.json").read_text(encoding="utf-8"))

_SHARED = {"passrate": _PASSRATE, "subj_vocab": _SUBJ_VOCAB,
           "passrate_row": _PASSRATE_ROW}

# ---------------------------------------------------------------------------------
# families (sequential load so encoders don't contend for GPU memory)
# ---------------------------------------------------------------------------------
_FAM_NAMES = list(_STACK_TOP["families"])          # e.g. ["qwen","nemotron","lgai"]
_FAMS: dict[str, dict] = {}
for _name in _FAM_NAMES:
    _t0 = time.time()
    _rt = fc.FamilyRuntime(HERE / _name, _SHARED)
    _enc = Encoder(_rt.meta["encoder"], HERE)
    _FAMS[_name] = {"rt": _rt, "enc": _enc,
                    "emb_cache": {}, "pending": [], "pending_keys": set()}
    LOG.info("family %s ready in %.1fs", _name, time.time() - _t0)

_L3_W = [float(_STACK_TOP["weights"][f]) for f in _FAM_NAMES]
_L3_B = float(_STACK_TOP["bias"])


# ---------------------------------------------------------------------------------
# streamed-flush encoder queues (trc5 streamed_flush_v1 semantics)
# ---------------------------------------------------------------------------------
def _embed_into_cache(fam: dict, batch: list[tuple[str, str, str, str]]) -> None:
    """batch: list of (item_key, benchmark, condition_norm, item_content)."""
    texts = [fc.item_text_for(b, c, ic,
                              fam["rt"].meta["encoder"].get("passage_prefix", ""))
             for (_k, b, c, ic) in batch]
    vecs = fam["enc"].embed(texts)
    for (k, *_rest), v in zip(batch, vecs):
        fam["emb_cache"][k] = v
        fam["pending_keys"].discard(k)


def _enqueue_for_batch(*, benchmark, condition, subject_content, item_content):
    """Fan out to every family's queue; flush at most one batch per family per call."""
    b = str(benchmark or "")
    c = fc.normalize_condition(condition)
    ic = str(item_content or "")
    k = fc.item_key_for(b, c, ic)
    for name, fam in _FAMS.items():
        try:
            if k not in fam["emb_cache"] and k not in fam["pending_keys"]:
                fam["pending"].append((k, b, c, ic))
                fam["pending_keys"].add(k)
            bs = fam["enc"].batch_size
            if len(fam["pending"]) >= bs:
                batch, fam["pending"] = fam["pending"][:bs], fam["pending"][bs:]
                _embed_into_cache(fam, batch)
        except Exception:
            LOG.exception("enqueue failed for family %s; continuing", name)


def _flush_pending(fam: dict) -> None:
    while fam["pending"]:
        bs = fam["enc"].batch_size
        batch, fam["pending"] = fam["pending"][:bs], fam["pending"][bs:]
        _embed_into_cache(fam, batch)


def _get_emb(fam: dict, item_key: str, b: str, c: str, ic: str) -> np.ndarray:
    v = fam["emb_cache"].get(item_key)
    if v is None:
        _embed_into_cache(fam, [(item_key, b, c, ic)])
        v = fam["emb_cache"][item_key]
    return v


# ---------------------------------------------------------------------------------
# trc5 calibrator: type-conditional partial-pool intercept (ported exactly)
# ---------------------------------------------------------------------------------
_RIDGE_LAMBDA_GLOBAL = 20.0
_RIDGE_LAMBDA_BC = 20.0
_RIDGE_LAMBDA_TYPE = 10.0


def _safe_logit(p: float) -> float:
    p = float(min(max(p, EPS), 1.0 - EPS))
    return math.log(p / (1.0 - p))


def _safe_sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-min(z, 50.0))
        p = 1.0 / (1.0 + e)
    else:
        e = math.exp(max(z, -50.0))
        p = e / (1.0 + e)
    return float(min(max(p, EPS), 1.0 - EPS))


def _fit_intercept_ridge(ps, ys, *, target_b=0.0, ridge):
    if not ps:
        return float(target_b)
    zs = [_safe_logit(p) for p in ps]
    b = float(target_b)
    for _ in range(80):
        g = 2.0 * ridge * (b - target_b)
        h = 2.0 * ridge
        for z, y in zip(zs, ys):
            q = _safe_sigmoid(z + b)
            g += q - y
            h += q * (1.0 - q)
        if h < 1e-9:
            break
        step = g / h
        b -= step
        if abs(step) < 1e-8:
            break
    return float(min(max(b, -5.0), 5.0))


class _Calibrator:
    def __init__(self):
        self.b_global = 0.0
        self.delta_type = 0.0
        self.per_bc: dict[str, float] = {}

    def fit_from_labeled(self, labeled) -> None:
        ps, ys, bcs, is_new = [], [], [], []
        for row in labeled or []:
            try:
                y = float(row.get("label"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(y):
                continue
            b = str(row.get("benchmark") or "")
            c = fc.normalize_condition(row.get("condition"))
            p = _predict_uncalibrated(b, c, str(row.get("subject_content") or ""),
                                      str(row.get("item_content") or ""))
            bc_key = f"{b}::{c}"
            ps.append(p)
            ys.append(y)
            bcs.append(bc_key)
            is_new.append(0.0 if bc_key in _BC_TO_ID else 1.0)
        if not ps:
            return
        # Stage 1: global intercept
        self.b_global = _fit_intercept_ridge(ps, ys, target_b=0.0,
                                             ridge=_RIDGE_LAMBDA_GLOBAL)
        # Stage 2: NEW-bc extra shift on pre-shifted probs
        new_ps = [_safe_sigmoid(_safe_logit(p) + self.b_global)
                  for p, n in zip(ps, is_new) if n > 0]
        new_ys = [y for y, n in zip(ys, is_new) if n > 0]
        self.delta_type = _fit_intercept_ridge(new_ps, new_ys, target_b=0.0,
                                               ridge=_RIDGE_LAMBDA_TYPE) if new_ps else 0.0
        # Stage 3: re-fit global on rows pre-shifted by delta_type*is_new
        ps3 = [_safe_sigmoid(_safe_logit(p) + self.delta_type * n)
               for p, n in zip(ps, is_new)]
        self.b_global = _fit_intercept_ridge(ps3, ys, target_b=0.0,
                                             ridge=_RIDGE_LAMBDA_GLOBAL)
        # Stage 4: per-bc full intercepts shrunk toward b_global + delta_type*is_new
        buckets: dict[str, list[int]] = {}
        for i, k in enumerate(bcs):
            buckets.setdefault(k, []).append(i)
        self.per_bc = {}
        for k, idxs in buckets.items():
            tgt = self.b_global + self.delta_type * is_new[idxs[0]]
            self.per_bc[k] = _fit_intercept_ridge(
                [ps[i] for i in idxs], [ys[i] for i in idxs],
                target_b=tgt, ridge=_RIDGE_LAMBDA_BC)

    def apply(self, p: float, bc_key: str) -> float:
        try:
            if bc_key in self.per_bc:
                b = self.per_bc[bc_key]
            elif bc_key not in _BC_TO_ID:
                b = self.b_global + self.delta_type
            else:
                b = self.b_global
            q = _safe_sigmoid(_safe_logit(p) + b)
            return q if math.isfinite(q) else DEFAULT_PROB
        except Exception:
            return DEFAULT_PROB


_CALIBRATOR = _Calibrator()
_LABELED_FP = None
_PROB_CACHE: dict[tuple[str, str], float] = {}


def _labeled_fingerprint(labeled):
    if not labeled:
        return None
    rows = []
    for r in labeled:
        try:
            rows.append((str(r.get("benchmark") or ""), str(r.get("condition") or ""),
                         stable_sha256(str(r.get("subject_content") or "")),
                         stable_sha256(str(r.get("item_content") or "")),
                         float(r.get("label", float("nan")))))
        except Exception:
            continue
    return tuple(sorted(rows))


# ---------------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------------
def _predict_uncalibrated(b: str, c: str, subject_content: str, item_content: str) -> float:
    item_key = fc.item_key_for(b, c, item_content)
    subject_key = fc.subject_key_for(subject_content)
    probs, weights = [], []
    for name, w in zip(_FAM_NAMES, _L3_W):
        fam = _FAMS[name]
        try:
            emb = _get_emb(fam, item_key, b, c, item_content)
            p = fam["rt"].predict_pair(subject_key, item_key, emb)
        except Exception:
            LOG.exception("family %s failed; dropping from blend", name)
            continue
        if not (p == p):
            continue
        probs.append(float(min(max(p, EPS), 1.0 - EPS)))
        weights.append(w)
    if not probs:
        return DEFAULT_PROB
    z = _L3_B + sum(w * _safe_logit(p) for p, w in zip(probs, weights))
    z = float(min(max(z, -_LOGIT_CLAMP), _LOGIT_CLAMP))
    return _safe_sigmoid(z)


def predict(input: dict, labeled=None) -> float:
    global _LABELED_FP, _CALIBRATOR
    try:
        for fam in _FAMS.values():
            try:
                _flush_pending(fam)
            except Exception:
                LOG.exception("flush failed; continuing")
        b = str(input.get("benchmark") or "")
        c = fc.normalize_condition(input.get("condition"))
        subject_content = str(input.get("subject_content") or "")
        item_content = str(input.get("item_content") or "")
        fp = _labeled_fingerprint(labeled)
        if fp is not None and fp != _LABELED_FP:
            _LABELED_FP = fp
            _PROB_CACHE.clear()
            cal = _Calibrator()
            cal.fit_from_labeled(labeled)
            _CALIBRATOR = cal
            LOG.info("calibrator refit on %d labeled rows: b_global=%.3f delta=%.3f "
                     "per_bc=%d", len(labeled), cal.b_global, cal.delta_type,
                     len(cal.per_bc))
        cache_key = (fc.item_key_for(b, c, item_content),
                     fc.subject_key_for(subject_content))
        hit = _PROB_CACHE.get(cache_key)
        if hit is not None:
            return float(hit)
        p = _predict_uncalibrated(b, c, subject_content, item_content)
        q = _CALIBRATOR.apply(p, f"{b}::{c}")
        q = float(min(max(q, EPS), 1.0 - EPS))
        _PROB_CACHE[cache_key] = q
        return q
    except Exception:
        LOG.exception("predict failed; returning DEFAULT_PROB")
        return float(DEFAULT_PROB)
