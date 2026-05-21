"""Behavioral smoke test for the new benchmark-novelty labeling.py.

Extracts labeling.py from the shipped bundle and stubs the model.py
imports it depends on (_enqueue_for_batch, _BC_TO_ID, _SUBJECT_TO_ID,
normalize_condition, stable_sha256).  Verifies:

  1. Acquisition score for a known benchmark + known subject is LOW.
  2. Acquisition score for a NEW benchmark + known subject is HIGH (>1000).
  3. Acquisition score for a NEW benchmark + UNKNOWN subject is MEDIUM
     (>1000 but < the known-subject case).
  4. Acquisition score for a known benchmark + unknown subject is LOW.
  5. _enqueue_for_batch is invoked exactly once per call (the streamed-
     encoder pipeline depends on this).
  6. If model.py imports fail, the score gracefully falls back to 0.0.
  7. Score difference between top and bottom is large enough that the
     platform's top-K-per-category sampler will reliably pick the
     new-benchmark + anchor-model candidates.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import tempfile
import types
import zipfile
from pathlib import Path

OUT_ZIP = Path(r"C:/Users/benja/Downloads/submission/submission_streamed_encoder_nn_perbc_cal.zip")


def _stub_module(
    known_bc_keys,
    known_subject_keys,
    enqueue_log,
    *,
    bc_counts=None,
    subj_counts=None,
):
    """Build a fake `model` module with the symbols labeling.py imports.

    If ``bc_counts`` and ``subj_counts`` are passed, the stub also exposes
    ``_N_TRAIN_PER_BC`` and ``_N_TRAIN_PER_SUBJECT`` so the graded path
    inside labeling.py kicks in.
    """

    def normalize_condition(c):
        return str(c or "none")

    def stable_sha256(s):
        return hashlib.sha256(str(s).encode("utf-8")).hexdigest()

    bc_to_id = {k: i + 1 for i, k in enumerate(sorted(known_bc_keys))}
    subject_to_id = {stable_sha256(s): i + 1 for i, s in enumerate(sorted(known_subject_keys))}

    def _enqueue_for_batch(**kwargs):
        enqueue_log.append(kwargs)

    mod = types.ModuleType("model")
    mod._BC_TO_ID = bc_to_id
    mod._SUBJECT_TO_ID = subject_to_id
    mod.normalize_condition = normalize_condition
    mod.stable_sha256 = stable_sha256
    mod._enqueue_for_batch = _enqueue_for_batch
    if bc_counts is not None:
        mod._N_TRAIN_PER_BC = dict(bc_counts)
    if subj_counts is not None:
        mod._N_TRAIN_PER_SUBJECT = {stable_sha256(k): v for k, v in subj_counts.items()}
    return mod


def _load_labeling_with_stub(model_stub):
    """Extract labeling.py from the bundle, register model_stub, import."""
    sys.modules["model"] = model_stub
    # Use a temp dir so we can import labeling.py as a real module.
    with zipfile.ZipFile(OUT_ZIP, "r") as zf:
        src = zf.read("labeling.py").decode("utf-8")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "labeling.py"
        path.write_text(src, encoding="utf-8")
        sys.path.insert(0, td)
        if "labeling" in sys.modules:
            del sys.modules["labeling"]
        mod = importlib.import_module("labeling")
        sys.path.pop(0)
        return mod


def main() -> int:
    known_bc = ["mmlu::none", "gsm8k::none", "humaneval::cot"]
    known_subjects = ["gpt-4", "llama-3-70b", "claude-3-opus", "mistral-7b"]
    new_bc = "secret_benchmark_v2::none"
    new_subject = "unknown_local_llm"

    # -------- BINARY mode (no counts shipped) --------
    print("=" * 60)
    print("BINARY mode (legacy bundles without train counts)")
    print("=" * 60)
    enqueue_log = []
    stub = _stub_module(known_bc, known_subjects, enqueue_log)
    labeling = _load_labeling_with_stub(stub)

    def score(benchmark, condition, subject, item, ll=labeling):
        return ll.acquisition_function({
            "benchmark": benchmark,
            "condition": condition,
            "subject_content": subject,
            "item_content": item,
        })

    s_known_known = score("mmlu", "none", "gpt-4", "what is 2+2?")
    s_new_known = score("secret_benchmark_v2", "none", "gpt-4", "what is 2+2?")
    s_new_unknown = score("secret_benchmark_v2", "none", "unknown_local_llm", "what is 2+2?")
    s_known_unknown = score("mmlu", "none", "unknown_local_llm", "what is 2+2?")

    print(
        "[scores]"
        + "\n  known_bc + known_subj   = {:+10.4f}".format(s_known_known)
        + "\n  NEW_bc   + known_subj   = {:+10.4f}".format(s_new_known)
        + "\n  NEW_bc   + UNKNOWN_subj = {:+10.4f}".format(s_new_unknown)
        + "\n  known_bc + UNKNOWN_subj = {:+10.4f}".format(s_known_unknown)
    )

    assert s_new_known > 1000.0, "new bc + anchor model should be > 1000"
    assert s_new_unknown > 1000.0, "new bc alone should still be > 1000"
    assert s_new_known > s_new_unknown, "anchor model should outrank unknown model within new bc"
    assert s_known_known < 100.0, "known bc + anchor model should be small"
    assert s_known_known > s_known_unknown, "anchor model should outrank unknown model within known bc"
    assert s_new_unknown - s_known_known > 900.0, "novelty band gap should be large"
    assert len(enqueue_log) == 4, "expected 4 enqueue calls, got " + str(len(enqueue_log))
    print("[OK] BINARY mode passes\n")

    # -------- GRADED mode (counts shipped) --------
    print("=" * 60)
    print("GRADED mode (re-exported bundles WITH train counts)")
    print("=" * 60)
    # 4 known benchmarks with very different training intensities; one
    # is very rare so it should still score nearly as high as a truly
    # unseen benchmark.
    bc_counts = {
        "mmlu::none": 5000,        # very popular -> low novelty
        "gsm8k::none": 500,        # medium
        "humaneval::cot": 50,      # rare
        "very_rare_bench::none": 1,  # extremely rare -> high novelty
    }
    subj_counts = {
        "gpt-4": 4000,             # max anchoring
        "llama-3-70b": 800,        # high
        "claude-3-opus": 400,      # medium
        "mistral-7b": 20,          # low but known
    }
    # Provide _BC_TO_ID for the rare bench so it counts as "known" with low count.
    known_bc_graded = list(bc_counts.keys())
    enqueue_log2 = []
    stub_g = _stub_module(
        known_bc_graded,
        list(subj_counts.keys()),
        enqueue_log2,
        bc_counts=bc_counts,
        subj_counts=subj_counts,
    )
    labeling_g = _load_labeling_with_stub(stub_g)

    def score_g(b, c, s, it, ll=labeling_g):
        return ll.acquisition_function({
            "benchmark": b, "condition": c,
            "subject_content": s, "item_content": it,
        })

    g_popular_top = score_g("mmlu", "none", "gpt-4", "Q1")
    g_popular_low = score_g("mmlu", "none", "mistral-7b", "Q1")
    g_rare_top = score_g("very_rare_bench", "none", "gpt-4", "Q1")
    g_unseen_top = score_g("totally_unseen_bench", "none", "gpt-4", "Q1")
    g_unseen_unknown = score_g("totally_unseen_bench", "none", "unknown_llm", "Q1")
    g_medium_medium = score_g("gsm8k", "none", "claude-3-opus", "Q1")

    print(
        "[graded scores]"
        + "\n  popular bc  + top model     = {:+10.4f}".format(g_popular_top)
        + "\n  popular bc  + low-anchor    = {:+10.4f}".format(g_popular_low)
        + "\n  very rare   + top model     = {:+10.4f}".format(g_rare_top)
        + "\n  UNSEEN bc   + top model     = {:+10.4f}".format(g_unseen_top)
        + "\n  UNSEEN bc   + UNKNOWN model = {:+10.4f}".format(g_unseen_unknown)
        + "\n  medium bc   + medium model  = {:+10.4f}".format(g_medium_medium)
    )

    # Graded ordering:
    # (a) UNSEEN benchmark (novelty=1) should outrank a known-but-rare benchmark.
    assert g_unseen_top > g_rare_top, "unseen bc must outrank rare-known bc"
    # (b) Rare-known should outrank popular.
    assert g_rare_top > g_popular_top, "rare-known bc should beat popular bc"
    # (c) Anchor strength matters within a tier (popular bc).
    assert g_popular_top > g_popular_low, "anchor strength must order within tier"
    # (d) Anchor strength still matters in the unseen tier.
    assert g_unseen_top > g_unseen_unknown, "anchor model must outrank unknown in unseen tier"
    # (e) The popular tier should be substantially lower than the unseen tier
    #     so the platform's top-K-per-category will reliably pick unseen rows.
    assert g_unseen_unknown - g_popular_top > 100.0, (
        "novelty band gap should still dominate (>100) even in graded mode"
    )
    print("[OK] GRADED mode passes\n")

    # Fallback path: when model.py fails to import, score should be 0.0.
    sys.modules.pop("model", None)
    sys.modules.pop("labeling", None)

    class FailingFinder:
        def find_module(self, name, path=None):
            if name == "model":
                return self
            return None

        def load_module(self, name):
            raise ImportError("fake")

    sys.meta_path.insert(0, FailingFinder())
    try:
        with zipfile.ZipFile(OUT_ZIP, "r") as zf:
            src = zf.read("labeling.py").decode("utf-8")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "labeling.py"
            path.write_text(src, encoding="utf-8")
            sys.path.insert(0, td)
            labeling2 = importlib.import_module("labeling")
            sys.path.pop(0)
        s_fallback = labeling2.acquisition_function({
            "benchmark": "anything",
            "condition": "none",
            "subject_content": "anything",
            "item_content": "anything",
        })
        assert s_fallback == 0.0, "fallback score should be 0.0 when model import fails"
        print("[OK] fallback to 0.0 when model.py import fails (got {})".format(s_fallback))
    finally:
        sys.meta_path.pop(0)

    print("[OK] all labeling.py behavioral tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
