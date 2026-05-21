"""Behavioral smoke test for the dual-pool stratified labeling.py.

Extracts labeling.py from the shipped bundle and stubs the model.py
imports it depends on (_enqueue_for_batch, _BC_TO_ID, _SUBJECT_TO_ID,
normalize_condition, stable_sha256).  Verifies:

  1. Each item is deterministically routed to either POOL A (new-bc
     prioritized) or POOL B (known-bc prioritized) via a stable hash
     of ``item_content``.
  2. In POOL A: a new-bc candidate scores HIGH (>=1000); a known-bc
     candidate scores LOW (<100).
  3. In POOL B: a known-bc candidate scores HIGH (>=1000); a new-bc
     candidate scores LOW (<100).
  4. Within a pool, anchoring + tiebreak order subjects (anchored
     subjects beat unknown ones).
  5. Across many random items, the empirical fraction routed to
     POOL A matches ``_FRACTION_NEW_POOL`` (0.95) within tolerance.
  6. ``_enqueue_for_batch`` fires exactly once per call.
  7. If model.py imports fail, score gracefully falls back to 0.0.
  8. Both BINARY (no train counts) and GRADED (with counts) paths
     route correctly.
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


def _find_item_in_pool(labeling, target_new_pool: bool, *, prefix: str) -> str:
    """Return an item_content string that the labeling module routes to
    ``new_pool == target_new_pool``.  Just enumerates suffixes until a
    match is found (the hash is uniform, so this terminates fast)."""
    for i in range(10_000):
        candidate = "{}-{}".format(prefix, i)
        if labeling._item_in_new_pool(candidate) == target_new_pool:
            return candidate
    raise RuntimeError("could not find a suitable item; hash is degenerate?")


def main() -> int:
    known_bc = ["mmlu::none", "gsm8k::none", "humaneval::cot"]
    known_subjects = ["gpt-4", "llama-3-70b", "claude-3-opus", "mistral-7b"]

    # ---------------------------------------------------------------------
    # BINARY mode (no train counts shipped)
    # ---------------------------------------------------------------------
    print("=" * 60)
    print("BINARY mode (bundles without train counts)")
    print("=" * 60)
    enqueue_log = []
    stub = _stub_module(known_bc, known_subjects, enqueue_log)
    labeling = _load_labeling_with_stub(stub)

    assert hasattr(labeling, "_FRACTION_NEW_POOL"), "labeling.py must expose _FRACTION_NEW_POOL"
    assert hasattr(labeling, "_item_in_new_pool"), "labeling.py must expose _item_in_new_pool"
    assert abs(labeling._FRACTION_NEW_POOL - 0.95) < 1e-9, (
        "tuned fraction must be 0.95, got " + str(labeling._FRACTION_NEW_POOL)
    )

    item_in_A = _find_item_in_pool(labeling, target_new_pool=True, prefix="poolA")
    item_in_B = _find_item_in_pool(labeling, target_new_pool=False, prefix="poolB")

    def score(benchmark, condition, subject, item, ll=labeling):
        return ll.acquisition_function({
            "benchmark": benchmark,
            "condition": condition,
            "subject_content": subject,
            "item_content": item,
        })

    # Pool A behavior: new bcs win, known bcs lose.
    s_A_new_known_subj = score("secret_new_bench", "none", "gpt-4", item_in_A)
    s_A_new_unknown_subj = score("secret_new_bench", "none", "unknown_llm", item_in_A)
    s_A_known_known_subj = score("mmlu", "none", "gpt-4", item_in_A)
    # Pool B behavior: known bcs win, new bcs lose.
    s_B_known_known_subj = score("mmlu", "none", "gpt-4", item_in_B)
    s_B_new_known_subj = score("secret_new_bench", "none", "gpt-4", item_in_B)

    print("[POOL A (new-prioritized)]")
    print("  new_bc   + known_subj   = {:+10.4f}".format(s_A_new_known_subj))
    print("  new_bc   + UNKNOWN_subj = {:+10.4f}".format(s_A_new_unknown_subj))
    print("  known_bc + known_subj   = {:+10.4f}".format(s_A_known_known_subj))
    print("[POOL B (known-prioritized)]")
    print("  known_bc + known_subj   = {:+10.4f}".format(s_B_known_known_subj))
    print("  new_bc   + known_subj   = {:+10.4f}".format(s_B_new_known_subj))

    assert s_A_new_known_subj >= 1000.0, "POOL A new-bc must score >=1000"
    assert s_A_new_known_subj > s_A_new_unknown_subj, (
        "anchor strength must order within POOL A new-bc tier"
    )
    assert s_A_known_known_subj < 100.0, "POOL A known-bc must score <100 (deprioritized)"
    assert s_B_known_known_subj >= 1000.0, "POOL B known-bc must score >=1000"
    assert s_B_new_known_subj < 100.0, "POOL B new-bc must score <100 (deprioritized)"

    # _enqueue_for_batch should have fired on each scoring call.
    assert len(enqueue_log) == 5, (
        "expected 5 enqueue calls, got " + str(len(enqueue_log))
    )
    print("[OK] BINARY mode pool routing correct\n")

    # ---------------------------------------------------------------------
    # Empirical pool fraction across 5000 random items
    # ---------------------------------------------------------------------
    n_in_A = 0
    N = 5000
    for i in range(N):
        if labeling._item_in_new_pool("random-item-{}".format(i)):
            n_in_A += 1
    frac = n_in_A / N
    print("[empirical pool fraction] N={}, in POOL A = {} ({:.2%})".format(N, n_in_A, frac))
    assert abs(frac - 0.95) < 0.02, (
        "empirical fraction must match _FRACTION_NEW_POOL=0.95 within 0.02; got {}".format(frac)
    )
    print("[OK] pool fraction matches target\n")

    # ---------------------------------------------------------------------
    # GRADED mode (train counts shipped)
    # ---------------------------------------------------------------------
    print("=" * 60)
    print("GRADED mode (bundles WITH train counts)")
    print("=" * 60)
    bc_counts = {
        "mmlu::none": 5000,
        "gsm8k::none": 500,
        "humaneval::cot": 50,
    }
    subj_counts = {
        "gpt-4": 4000,
        "llama-3-70b": 800,
        "claude-3-opus": 400,
        "mistral-7b": 20,
    }
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

    g_item_A = _find_item_in_pool(labeling_g, target_new_pool=True, prefix="gA")
    g_item_B = _find_item_in_pool(labeling_g, target_new_pool=False, prefix="gB")

    def score_g(b, c, s, it, ll=labeling_g):
        return ll.acquisition_function({
            "benchmark": b, "condition": c,
            "subject_content": s, "item_content": it,
        })

    # In POOL A: an unseen benchmark scores HIGH; a known-but-rare scores LOW.
    g_A_unseen_top = score_g("totally_unseen_bench", "none", "gpt-4", g_item_A)
    g_A_rare_known = score_g("humaneval", "cot", "gpt-4", g_item_A)
    g_A_popular_known = score_g("mmlu", "none", "gpt-4", g_item_A)
    # In POOL B: a known benchmark scores HIGH; an unseen scores LOW.
    g_B_unseen = score_g("totally_unseen_bench", "none", "gpt-4", g_item_B)
    g_B_rare_known = score_g("humaneval", "cot", "gpt-4", g_item_B)
    g_B_popular_known = score_g("mmlu", "none", "gpt-4", g_item_B)
    # Anchor strength within tier:
    g_A_unseen_low_anchor = score_g("totally_unseen_bench", "none", "mistral-7b", g_item_A)

    print("[GRADED POOL A]")
    print("  unseen_bc  + top model = {:+10.4f}".format(g_A_unseen_top))
    print("  rare_known + top model = {:+10.4f}".format(g_A_rare_known))
    print("  popular_kn + top model = {:+10.4f}".format(g_A_popular_known))
    print("[GRADED POOL B]")
    print("  unseen_bc  + top model = {:+10.4f}".format(g_B_unseen))
    print("  rare_known + top model = {:+10.4f}".format(g_B_rare_known))
    print("  popular_kn + top model = {:+10.4f}".format(g_B_popular_known))
    print("[anchor ordering in pool A]")
    print("  unseen_bc  + low model = {:+10.4f}".format(g_A_unseen_low_anchor))

    assert g_A_unseen_top >= 1000.0, "POOL A unseen-bc must score >=1000"
    assert g_A_rare_known < 100.0, "POOL A known-bc (rare or not) must score <100"
    assert g_A_popular_known < 100.0, "POOL A popular-known must score <100"
    assert g_B_rare_known >= 1000.0, "POOL B rare-known must score >=1000"
    assert g_B_popular_known >= 1000.0, "POOL B popular-known must score >=1000"
    assert g_B_unseen < 100.0, "POOL B unseen-bc must score <100"
    assert g_A_unseen_top > g_A_unseen_low_anchor, (
        "anchor strength must order within POOL A unseen tier"
    )
    print("[OK] GRADED mode pool routing correct\n")

    # ---------------------------------------------------------------------
    # Fallback: when model.py fails to import, score should be 0.0.
    # ---------------------------------------------------------------------
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
