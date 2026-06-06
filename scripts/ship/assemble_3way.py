#!/usr/bin/env python3
"""Assemble the 3-embedding Codabench submission in the PROVEN structure.

  qwen8b (Qwen/Qwen3-Embedding-8B)
  nemotron (nvidia/llama-embed-nemotron-8b)
  lgai (annamodels/LGAI-Embedding-Preview)

combined by a top-level logit-FWLS / geometric-mean-of-odds combiner that
importlib-loads each subdir's ``submodel.py``.

This is a standalone, self-contained Colab cell script. It needs NOTHING from
this repo at runtime (the produced bundle ships only torch/transformers/numpy/
safetensors/huggingface_hub/tokenizers/sentencepiece + stdlib). It only reads
the three already-trained per-family bundles (the uploaded zips OR extracted
dirs) and writes a single submission zip.

--------------------------------------------------------------------------
WHAT THIS REPRODUCES (and how it differs from the repo's build_ensemble)
--------------------------------------------------------------------------
The PROVEN bundle (`/tmp/fwls_inspect/ensemble_3way_logit_fwls`, == the sent
`ensemble_3way_logit_fwls.zip`) has this layout, with subdirs AT THE BUNDLE
ROOT, each holding a ``submodel.py`` (NOT a ``model.py``):

    model.py            <- top-level combiner (this script writes it)
    models.txt          <- the 3 encoder slugs
    labeling.py         <- reused from a sent bundle (acquisition fn)
    qwen8b/   submodel.py + artifacts/
    nemotron/ submodel.py + artifacts/
    lgai/     submodel.py + artifacts/

This DIFFERS from `scripts/build_ensemble_submission.py::build_ensemble`,
which stages submodels under `submodels/<name>/` and loads each subdir's
`model.py`. We deliberately do NOT use that path -- we replicate the proven
top-level combiner verbatim (parametrised only by subdir names + fixed
equal-weight logit averaging, with an optional val-fit weight hook).

--------------------------------------------------------------------------
INPUT NORMALISATION (the families ship in two shapes)
--------------------------------------------------------------------------
  * qwen8b ships as a SUBMODEL bundle: top-level `submodel.py` + `artifacts/`.
    -> copied as-is into `qwen8b/`.
  * nemotron_trc_v5 and LGAI_fixed ship as STANDALONE single-model bundles:
    top-level `model.py` + `models.txt` + `labeling.py` + `artifacts/`.
    Their module-level API (predict, _enqueue_for_batch, _BC_TO_ID,
    _SUBJECT_TO_ID, _N_TRAIN_PER_BC, _N_TRAIN_PER_SUBJECT, normalize_condition,
    stable_sha256, DEFAULT_PROB, EPS) is EXACTLY what the combiner needs, so we
    rename `model.py` -> `submodel.py` and copy `artifacts/`. We drop their own
    top-level `models.txt`/`labeling.py` (the combiner supplies the bundle-root
    ones). All three use the SAME proven runtime (`streamed_flush_v1+perbc_cal`,
    `default_calibrator={'kind':'identity'}`), so the calibrator pattern is
    already correct -- this assembler does not touch heads or calibrators.

--------------------------------------------------------------------------
COMBINER (NO honest OOF)
--------------------------------------------------------------------------
  * DEFAULT: FIXED equal logit weights == geometric mean of odds == the proven
    model.py. No data needed. This is what ships unless you opt in.
  * OPTIONAL: fit per-model logit weights on a SINGLE held-out val split, if you
    pass --val-preds (a .npz/.json of per-model probs + true labels). This is a
    single-split FWLS, NOT k-fold OOF; it is mild and clipped. See
    `fit_logit_weights_from_val`. The produced model.py reads the weights from a
    sibling `ensemble_weights.json`; absent that file it falls back to equal
    weights, so the bundle is robust even if the json is stripped.

Usage (Colab cell or CLI)
-------------------------
    python assemble_3way.py \
        --qwen   /content/ship_bundles/qwen8b.zip \
        --nemo   /content/ship_bundles/nemotron_trc_v5.zip \
        --lgai   /content/ship_bundles/LGAI_fixed.zip \
        --out    /content/ensemble_3way_logit_fwls.zip
        [--val-preds /content/val_preds.npz]   # optional single-split FWLS
        [--labeling-from lgai]                  # which bundle's labeling.py
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence

LOG = logging.getLogger("assemble_3way")

# ---------------------------------------------------------------------------
# Canonical family spec. The order here == the order the combiner loads and
# (if used) the order val-preds weights are applied.
# ---------------------------------------------------------------------------
FAMILIES = [
    # (subdir, encoder slug, default source-key)
    ("qwen8b", "Qwen/Qwen3-Embedding-8B", "qwen"),
    ("nemotron", "nvidia/llama-embed-nemotron-8b", "nemo"),
    ("lgai", "annamodels/LGAI-Embedding-Preview", "lgai"),
]
SUBDIRS = [f[0] for f in FAMILIES]
ENCODER_SLUGS = [f[1] for f in FAMILIES]

ZIP_LIMIT_MB = 65.0

# Runtime import whitelist (HARD constraint). stdlib is allowed implicitly.
ALLOWED_THIRD_PARTY = {
    "torch", "numpy", "transformers", "safetensors",
    "huggingface_hub", "tokenizers", "sentencepiece",
}
# These must NEVER appear as a runtime import in any shipped .py.
FORBIDDEN = {
    "sklearn", "scikit_learn", "lightgbm", "xgboost", "catboost",
    "scipy", "pandas", "faiss", "joblib", "matplotlib", "seaborn",
    "cupy", "numba", "polars", "datasets",
}

# stdlib modules we expect to see in the shipped .py (informational only;
# the check below treats anything not in ALLOWED_THIRD_PARTY and not FORBIDDEN
# as a candidate stdlib import and verifies it resolves as stdlib).
_STDLIB_HINT = sys.stdlib_module_names if hasattr(sys, "stdlib_module_names") else set()


# ===========================================================================
# Top-level combiner model.py (PROVEN logit-average, parametrised on SUBDIRS)
# ===========================================================================
def _render_combiner_model_py(subdirs: Sequence[str], indexer_priority: Sequence[str]) -> str:
    """Render the bundle-root model.py.

    Identical in mechanics to the proven `/tmp/fwls_inspect/model.py`:
      * importlib-loads each `<subdir>/submodel.py`,
      * re-exports the indexer symbols labeling.py needs,
      * fans out `_enqueue_for_batch`,
      * combines `predict` via WEIGHTED logit average (geometric mean of odds
        when weights are equal).

    Added vs. the proven file: an optional weight vector read from a sibling
    `ensemble_weights.json` (absent -> equal weights == the proven behaviour).
    """
    subdirs_lit = repr(list(subdirs))
    indexer_lit = repr(list(indexer_priority))
    return f'''"""3-way ensemble combiner: Qwen3-Embedding-8B + llama-embed-nemotron-8b + LGAI.

Variant: logit-FWLS (geometric mean of odds; weighted if ensemble_weights.json present).

Each sub-submission lives untouched in its own subfolder and is loaded here as a
Python sub-module via importlib. Each sub-module does its own encoder + head +
calibrator setup against its own artifacts/. We never modify their code.

The combiner: (1) re-exports the indexer symbols labeling.py looks up on the
top-level `model` module; (2) fans out `_enqueue_for_batch` to all subs; (3)
`predict` = weighted logit average over the subs, then sigmoid. Default weights
are EQUAL (== geometric mean of odds == the proven FWLS combiner) because there
is no honest OOF; an optional `ensemble_weights.json` (single held-out val split)
overrides them.
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
del _os

import importlib.util
import json as _json
import logging
import math
import sys
import time
from pathlib import Path

LOG = logging.getLogger("ensemble3")
HERE = Path(__file__).resolve().parent

_SUB_NAMES = {subdirs_lit}
_INDEXER_PRIORITY = {indexer_lit}


def _load_submodel(subdir_name):
    submodel_path = HERE / subdir_name / "submodel.py"
    if not submodel_path.exists():
        raise FileNotFoundError("Ensemble sub-module not found: " + str(submodel_path))
    module_name = "_ensemble3_submodel_" + subdir_name
    spec = importlib.util.spec_from_file_location(module_name, submodel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not build importlib spec for " + str(submodel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    LOG.info("ensemble3: loading sub-module %s", submodel_path)
    t0 = time.time()
    spec.loader.exec_module(mod)
    LOG.info("ensemble3: loaded %s in %.1fs", module_name, time.time() - t0)
    return mod


# Load sub-modules sequentially so each loader's .to(_DEVICE) step doesn't
# contend for GPU memory with the previous one.
_SUBS = []
_LOADED_NAMES = []
for _nm in _SUB_NAMES:
    try:
        _SUBS.append(_load_submodel(_nm))
        _LOADED_NAMES.append(_nm)
    except Exception:  # noqa: BLE001
        LOG.exception("ensemble3: failed to load sub-module %s; continuing without it", _nm)
_SUB_NAMES = _LOADED_NAMES
if not _SUBS:
    raise RuntimeError("ensemble3: no sub-modules loaded; cannot build ensemble")


# ---------------------------------------------------------------------------
# Optional per-model logit weights (single held-out val split). Absent file or
# any parse failure -> equal weights (== geometric mean of odds, the proven
# combiner). Weights are keyed by subdir name and renormalised over the subs
# that actually loaded.
# ---------------------------------------------------------------------------
def _load_weights():
    wpath = HERE / "ensemble_weights.json"
    w = {{nm: 1.0 for nm in _SUB_NAMES}}
    if wpath.exists():
        try:
            raw = _json.loads(wpath.read_text(encoding="utf-8"))
            cand = raw.get("weights", raw) if isinstance(raw, dict) else {{}}
            picked = {{}}
            for nm in _SUB_NAMES:
                v = cand.get(nm)
                if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0:
                    picked[nm] = float(v)
            if len(picked) == len(_SUB_NAMES):
                w = picked
                LOG.info("ensemble3: loaded val-fit weights %s", w)
            else:
                LOG.warning("ensemble3: ensemble_weights.json incomplete; using equal weights")
        except Exception:  # noqa: BLE001
            LOG.exception("ensemble3: failed to parse ensemble_weights.json; equal weights")
    s = sum(w.values()) or 1.0
    return [w[nm] / s for nm in _SUB_NAMES]


_WEIGHTS = _load_weights()


# ---------------------------------------------------------------------------
# Re-export the symbols labeling.py looks up on the `model` module. We prefer
# the indexer source with the largest GRADED train-counts; fall back to BINARY.
# ---------------------------------------------------------------------------
def _select_indexer_source():
    order = [n for n in _INDEXER_PRIORITY if n in _SUB_NAMES] + \\
            [n for n in _SUB_NAMES if n not in _INDEXER_PRIORITY]
    name_to_sub = dict(zip(_SUB_NAMES, _SUBS))
    cands = [(n, name_to_sub[n]) for n in order]
    for name, sub in cands:  # GRADED preferred
        bc = getattr(sub, "_BC_TO_ID", None) or {{}}
        ntpb = getattr(sub, "_N_TRAIN_PER_BC", None) or {{}}
        if isinstance(bc, dict) and bc and isinstance(ntpb, dict) and ntpb:
            return name, sub, "GRADED"
    for name, sub in cands:  # BINARY fallback
        bc = getattr(sub, "_BC_TO_ID", None) or {{}}
        if isinstance(bc, dict) and bc:
            return name, sub, "BINARY"
    return cands[0][0], cands[0][1], "EMPTY"


_INDEXER_SRC_NAME, _INDEXER_SRC, _INDEXER_MODE = _select_indexer_source()
_BC_TO_ID = getattr(_INDEXER_SRC, "_BC_TO_ID", {{}}) or {{}}
_SUBJECT_TO_ID = getattr(_INDEXER_SRC, "_SUBJECT_TO_ID", {{}}) or {{}}
_N_TRAIN_PER_BC = getattr(_INDEXER_SRC, "_N_TRAIN_PER_BC", {{}}) or {{}}
_N_TRAIN_PER_SUBJECT = getattr(_INDEXER_SRC, "_N_TRAIN_PER_SUBJECT", {{}}) or {{}}
normalize_condition = getattr(_INDEXER_SRC, "normalize_condition", None)
stable_sha256 = getattr(_INDEXER_SRC, "stable_sha256", None)
for _sub in _SUBS:  # robust fallbacks for the two callables
    if normalize_condition is None:
        normalize_condition = getattr(_sub, "normalize_condition", None)
    if stable_sha256 is None:
        stable_sha256 = getattr(_sub, "stable_sha256", None)
LOG.info(
    "ensemble3: indexer src=%s mode=%s sizes BC=%d SUBJ=%d NTPB=%d NTPS=%d",
    _INDEXER_SRC_NAME, _INDEXER_MODE, len(_BC_TO_ID), len(_SUBJECT_TO_ID),
    len(_N_TRAIN_PER_BC), len(_N_TRAIN_PER_SUBJECT),
)
if _INDEXER_MODE != "GRADED":
    LOG.error("ensemble3: indexer mode %s (not GRADED); acquisition degrades, predict unaffected", _INDEXER_MODE)


def _enqueue_for_batch(*, benchmark, condition, subject_content, item_content):
    for sub, name in zip(_SUBS, _SUB_NAMES):
        try:
            sub._enqueue_for_batch(
                benchmark=benchmark, condition=condition,
                subject_content=subject_content, item_content=item_content,
            )
        except Exception:  # noqa: BLE001
            LOG.exception("ensemble3: %s._enqueue_for_batch failed; continuing", name)


DEFAULT_PROB = float(getattr(_SUBS[0], "DEFAULT_PROB", 0.5))
EPS = float(getattr(_SUBS[0], "EPS", 1e-6))
_LOGIT_CLAMP = 20.0


def _safe_logit(p):
    p = float(min(max(p, EPS), 1.0 - EPS))
    return math.log(p / (1.0 - p))


def _safe_sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        p = 1.0 / (1.0 + e)
    else:
        e = math.exp(z)
        p = e / (1.0 + e)
    return float(min(max(p, EPS), 1.0 - EPS))


def predict(input, labeled=None):
    """Weighted logit average over the sub-modules, then sigmoid.

    Equal weights == geometric mean of odds == the proven FWLS combiner.
    A sub-module that raises or returns NaN is dropped (its weight removed and
    the rest renormalised). All fail -> DEFAULT_PROB.
    """
    zs, ws = [], []
    for sub, name, w in zip(_SUBS, _SUB_NAMES, _WEIGHTS):
        try:
            p = float(sub.predict(input, labeled))
        except Exception:  # noqa: BLE001
            LOG.exception("ensemble3: %s.predict failed; skipping", name)
            continue
        if not (p == p):  # NaN
            LOG.warning("ensemble3: %s.predict returned NaN; skipping", name)
            continue
        p = float(min(max(p, EPS), 1.0 - EPS))
        zs.append(_safe_logit(p))
        ws.append(float(w))
    if not zs:
        return float(DEFAULT_PROB)
    wsum = sum(ws) or 1.0
    z_mean = sum(z * w for z, w in zip(zs, ws)) / wsum
    z_mean = float(min(max(z_mean, -_LOGIT_CLAMP), _LOGIT_CLAMP))
    return _safe_sigmoid(z_mean)
'''


# ===========================================================================
# Optional single-split FWLS weight fit (NO k-fold; mild, clipped)
# ===========================================================================
def fit_logit_weights_from_val(val_preds_path: str | Path) -> dict[str, float]:
    """Fit per-model logit weights on ONE held-out val split (no OOF).

    Input file (.npz or .json) must provide, in canonical FAMILY order:
      * either keys 'qwen8b'/'nemotron'/'lgai' each a length-N prob vector,
        or a 2-D array 'preds' of shape (N, 3) in FAMILY order;
      * 'y' (or 'labels'): length-N {0,1} ground truth.

    We do a tiny coordinate-free grid/logistic fit in LOGIT space using ONLY
    numpy (no scipy/sklearn -- runtime whitelist also constrains the trainer
    here for portability). The fit is a non-negative weighted logit average
    minimising val log-loss, found by multiplicative-weights coordinate ascent;
    weights are floored at 0.05 and renormalised so no model is dropped.

    Returns {subdir: weight}. CAVEAT: a single-split fit can overfit ~3 dof;
    keep it mild. If you do not trust the split, DON'T pass --val-preds and the
    bundle ships equal weights (the safe default).
    """
    import numpy as np  # local import: only needed when this hook is used

    p = Path(val_preds_path)
    if p.suffix.lower() == ".npz":
        d = np.load(p, allow_pickle=False)
        d = {k: d[k] for k in d.files}
    else:
        raw = json.loads(p.read_text(encoding="utf-8"))
        d = {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}

    if "preds" in d:
        P = np.asarray(d["preds"], dtype=np.float64)
        if P.shape[1] != len(SUBDIRS):
            raise ValueError(f"'preds' must be (N,{len(SUBDIRS)}) in FAMILY order; got {P.shape}")
    else:
        cols = []
        for sub in SUBDIRS:
            if sub not in d:
                raise ValueError(f"val-preds missing column for '{sub}' (and no 'preds' array)")
            cols.append(np.asarray(d[sub], dtype=np.float64).reshape(-1))
        P = np.stack(cols, axis=1)

    y = None
    for k in ("y", "labels", "label", "target"):
        if k in d:
            y = np.asarray(d[k], dtype=np.float64).reshape(-1)
            break
    if y is None:
        raise ValueError("val-preds must include ground truth under 'y'/'labels'")
    if y.shape[0] != P.shape[0]:
        raise ValueError(f"len(y)={y.shape[0]} != len(preds)={P.shape[0]}")

    eps = 1e-6
    P = np.clip(P, eps, 1.0 - eps)
    Z = np.log(P / (1.0 - P))  # (N, M) logits
    y = np.clip(y, 0.0, 1.0)

    def nll(w):
        w = np.asarray(w, dtype=np.float64)
        zc = Z @ w
        zc = np.clip(zc, -20.0, 20.0)
        pc = 1.0 / (1.0 + np.exp(-zc))
        pc = np.clip(pc, eps, 1.0 - eps)
        return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))

    # Multiplicative-weights coordinate ascent over a small grid; pure numpy.
    M = Z.shape[1]
    w = np.full(M, 1.0 / M, dtype=np.float64)
    best = nll(w)
    for _ in range(400):
        improved = False
        for j in range(M):
            for delta in (0.05, -0.05, 0.15, -0.15):
                cand = w.copy()
                cand[j] = max(cand[j] + delta, 0.0)
                s = cand.sum()
                if s <= 0:
                    continue
                cand = cand / s
                v = nll(cand)
                if v < best - 1e-9:
                    w, best = cand, v
                    improved = True
        if not improved:
            break
    # Floor + renormalise so no model is dropped entirely.
    w = np.maximum(w, 0.05)
    w = w / w.sum()
    out = {sub: float(wi) for sub, wi in zip(SUBDIRS, w)}
    LOG.info("fit_logit_weights_from_val: weights=%s  val_nll=%.5f", out, best)
    return out


# ===========================================================================
# Bundle normalisation: each input -> a directory with submodel.py + artifacts/
# ===========================================================================
def _extract_to(src: Path, dest: Path) -> Path:
    """Return a directory view of `src` (extract if it's a .zip)."""
    if src.is_dir():
        return src
    if src.is_file() and src.suffix.lower() == ".zip":
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src, "r") as zf:
            # zip-slip guard
            for member in zf.namelist():
                tgt = (dest / member).resolve()
                if not str(tgt).startswith(str(dest.resolve())):
                    raise RuntimeError(f"unsafe zip member path: {member!r}")
            zf.extractall(dest)
        return dest
    raise FileNotFoundError(f"{src} is neither a directory nor a .zip")


def _find_bundle_root(d: Path) -> Path:
    """Find the dir that actually holds artifacts/ + (submodel.py|model.py).

    Tolerates a single wrapping folder inside the zip.
    """
    def looks_like_root(x: Path) -> bool:
        has_art = (x / "artifacts").is_dir()
        has_code = (x / "submodel.py").is_file() or (x / "model.py").is_file()
        return has_art and has_code

    if looks_like_root(d):
        return d
    subs = [c for c in d.iterdir() if c.is_dir()]
    for c in subs:
        if looks_like_root(c):
            return c
    # last resort: any descendant within 2 levels
    for c in d.rglob("artifacts"):
        if c.is_dir() and looks_like_root(c.parent):
            return c.parent
    raise RuntimeError(f"could not locate a bundle root (artifacts/ + code) under {d}")


def _stage_family(src: Path, subdir: str, dest_root: Path) -> dict:
    """Copy one family into `dest_root/<subdir>/` as submodel.py + artifacts/.

    Handles both shapes:
      * SUBMODEL bundle: top-level submodel.py        -> copied as submodel.py
      * STANDALONE bundle: top-level model.py         -> renamed to submodel.py
    Drops any top-level models.txt / labeling.py from the family bundle (the
    combiner supplies the bundle-root ones).
    """
    with tempfile.TemporaryDirectory(prefix=f"stage_{subdir}_") as tmp:
        view = _extract_to(src, Path(tmp) / "x")
        root = _find_bundle_root(view)

        dst = dest_root / subdir
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

        # code file
        code_src = root / "submodel.py"
        code_kind = "submodel.py"
        if not code_src.is_file():
            code_src = root / "model.py"
            code_kind = "model.py->submodel.py"
        if not code_src.is_file():
            raise RuntimeError(f"{subdir}: no submodel.py or model.py under {root}")
        shutil.copy2(code_src, dst / "submodel.py")

        # artifacts/
        art_src = root / "artifacts"
        if not art_src.is_dir():
            raise RuntimeError(f"{subdir}: artifacts/ missing under {root}")
        shutil.copytree(art_src, dst / "artifacts")

        # any other sibling dirs the submodel may need (e.g. cache/) -- copy
        # extra top-level *files* that are not the dropped combiner pieces, and
        # extra *dirs* besides artifacts/. We KEEP these for safety.
        DROP = {"models.txt", "labeling.py", "model.py", "submodel.py"}
        extras = []
        for child in root.iterdir():
            if child.name in DROP or child.name == "artifacts":
                continue
            tgt = dst / child.name
            if child.is_dir():
                shutil.copytree(child, tgt)
                extras.append(child.name + "/")
            else:
                shutil.copy2(child, tgt)
                extras.append(child.name)

        # quick artifact inventory + calibrator sanity
        meta_path = dst / "artifacts" / "runtime_meta.json"
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
        return {
            "subdir": subdir,
            "code_kind": code_kind,
            "artifacts": sorted(p.name for p in (dst / "artifacts").iterdir()),
            "extras": extras,
            "encoder_model_id": meta.get("encoder_model_id"),
            "runtime_architecture": meta.get("runtime_architecture"),
            "default_calibrator": meta.get("default_calibrator"),
        }


def _pick_labeling_py(family_views: dict[str, Path], prefer: str) -> Path:
    """Return a labeling.py path, reused from a sent bundle (prefer one)."""
    order = [prefer] + [k for k in family_views if k != prefer]
    for key in order:
        view = family_views.get(key)
        if view is None:
            continue
        try:
            root = _find_bundle_root(view)
        except Exception:  # noqa: BLE001
            continue
        lp = root / "labeling.py"
        if lp.is_file():
            return lp
    raise RuntimeError(
        "no labeling.py found in any input bundle; supply one with --labeling-file"
    )


# ===========================================================================
# Static whitelist-only import check (over the WHOLE staged bundle)
# ===========================================================================
def _module_top(name: str) -> str:
    return name.split(".", 1)[0]


def _collect_imports(
    tree: ast.AST,
    *,
    top_level_only: bool,
) -> list[tuple[str, int, bool]]:
    """Walk `tree`; return (module, lineno, is_module_scope_unguarded).

    `is_module_scope_unguarded` is True iff the import executes
    unconditionally at module import time -- i.e. it is NOT nested in any
    function/class/try-handler. Those are the only imports that actually run
    when Codabench `importlib`-loads the submodel; lazy imports inside function
    bodies or `try:` blocks are how the proven bundle legitimately ships
    pandas/scipy/faiss (training/cache paths that never fire at runtime).
    """
    out: list[tuple[str, int, bool]] = []

    # Build a set of node ids that sit at module scope and outside try/except.
    # We do a manual descent tracking "guarded" context.
    def visit(node: ast.AST, module_scope: bool, guarded: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_module_scope = module_scope
            child_guarded = guarded
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child_module_scope = False
            if isinstance(child, ast.Try):
                # bodies inside Try are "guarded" (import errors handled)
                pass
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                mods: list[str] = []
                if isinstance(child, ast.Import):
                    mods = [a.name for a in child.names]
                else:
                    if child.level and child.level > 0:
                        mods = []  # relative -> intra-bundle, skip
                    elif child.module:
                        mods = [child.module]
                unguarded_module = module_scope and not guarded
                for m in mods:
                    out.append((m, getattr(child, "lineno", 0), unguarded_module))
            # descend; Try handler/body marks guarded for its subtree
            if isinstance(child, ast.Try):
                for sub in child.body:
                    visit(sub, child_module_scope, True)
                for h in child.handlers:
                    visit(h, child_module_scope, True)
                for sub in child.orelse:
                    visit(sub, child_module_scope, True)
                for sub in child.finalbody:
                    visit(sub, child_module_scope, guarded)
            else:
                visit(child, child_module_scope, child_guarded)

    visit(tree, True, False)
    if top_level_only:
        return [t for t in out if t[2]]
    return out


def check_imports_whitelist(stage_dir: Path) -> list[str]:
    """AST-scan every shipped .py; return a list of violation strings.

    A HARD violation = a module-scope, unguarded import (executes at
    `importlib` load time) of a module that is neither stdlib nor in
    ALLOWED_THIRD_PARTY. FORBIDDEN modules at module scope are flagged loudest.

    Lazy/guarded imports (inside a function body or a `try:` block) are NOT
    violations -- the proven bundle ships pandas/scipy/faiss/peft/langdetect
    exactly this way (training/cache paths that never run at Codabench runtime).
    Those are reported separately as INFO so the human can eyeball them.
    """
    violations: list[str] = []
    info_lazy: list[str] = []
    py_files = sorted(stage_dir.rglob("*.py"))
    local_names = {p.stem for p in py_files} | {"model", "submodel", "labeling"}
    for py in py_files:
        rel = py.relative_to(stage_dir)
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError as e:
            violations.append(f"{rel}: SYNTAX ERROR: {e}")
            continue
        for m, lineno, module_scope in _collect_imports(tree, top_level_only=False):
            top = _module_top(m)
            if not top or top in ALLOWED_THIRD_PARTY or top in local_names or top in _STDLIB_HINT:
                continue
            tag = "FORBIDDEN" if top in FORBIDDEN else "non-whitelisted"
            if module_scope:
                violations.append(f"{rel}:{lineno}: MODULE-SCOPE {tag} import '{m}'")
            else:
                info_lazy.append(f"{rel}:{lineno}: lazy/guarded {tag} import '{m}' (OK - not run at load)")
    if info_lazy:
        LOG.info(
            "whitelist check: %d lazy/guarded non-whitelisted imports (expected; "
            "match the proven bundle): %s", len(info_lazy), "; ".join(info_lazy[:8]) +
            (" ..." if len(info_lazy) > 8 else ""),
        )
    return violations


# ===========================================================================
# Zip + size audit
# ===========================================================================
def _zip_dir(stage_dir: Path, out_zip: Path) -> Path:
    out_zip = Path(out_zip).resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(stage_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(stage_dir).as_posix())
    return out_zip


def audit_zip_size(out_zip: Path) -> dict:
    size_mb = out_zip.stat().st_size / 1e6
    return {
        "zip": str(out_zip),
        "size_mb": round(size_mb, 3),
        "limit_mb": ZIP_LIMIT_MB,
        "ok": size_mb <= ZIP_LIMIT_MB,
    }


# ===========================================================================
# Top-level assemble
# ===========================================================================
def assemble(
    qwen: str | Path,
    nemo: str | Path,
    lgai: str | Path,
    out_zip: str | Path,
    *,
    val_preds: str | Path | None = None,
    labeling_from: str = "lgai",
    workspace: str | Path | None = None,
    strict: bool = True,
) -> dict:
    """Assemble the 3-way proven bundle. Returns a manifest dict."""
    src_by_key = {"qwen": Path(qwen), "nemo": Path(nemo), "lgai": Path(lgai)}
    for k, v in src_by_key.items():
        if not v.exists():
            raise FileNotFoundError(f"input '{k}' not found: {v}")

    ws = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix="assemble3_"))
    ws.mkdir(parents=True, exist_ok=True)
    stage = ws / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # 1. Stage each family -> <subdir>/submodel.py + artifacts/
    staged = []
    family_views: dict[str, Path] = {}
    for subdir, slug, srckey in FAMILIES:
        src = src_by_key[srckey]
        # keep an extracted view around for labeling.py selection
        view_dir = ws / f"view_{srckey}"
        view = _extract_to(src, view_dir) if src.suffix.lower() == ".zip" or not src.is_dir() else src
        family_views[srckey] = view
        info = _stage_family(src, subdir, stage)
        info["slug"] = slug
        staged.append(info)
        LOG.info("staged %s: %s", subdir, info)

    # 2. models.txt (3 encoder slugs, canonical order)
    (stage / "models.txt").write_text("\n".join(ENCODER_SLUGS) + "\n", encoding="utf-8")

    # 3. labeling.py reused from a sent bundle
    lp_src = _pick_labeling_py(family_views, prefer=labeling_from)
    shutil.copy2(lp_src, stage / "labeling.py")

    # 4. combiner model.py (indexer priority: qwen8b first -- largest train BC)
    indexer_priority = ["qwen8b", "nemotron", "lgai"]
    (stage / "model.py").write_text(
        _render_combiner_model_py(SUBDIRS, indexer_priority), encoding="utf-8"
    )

    # 5. NO-OOF combiner: default equal weights; optional single-split FWLS
    weights = None
    if val_preds is not None:
        weights = fit_logit_weights_from_val(val_preds)
        (stage / "ensemble_weights.json").write_text(
            json.dumps({"weights": weights, "source": "single_val_split_fwls"}, indent=2),
            encoding="utf-8",
        )

    # 6. ensemble_meta.json (provenance; not read at runtime by predict)
    ens_meta = {
        "structure": "proven_logit_fwls_3way",
        "subdirs": SUBDIRS,
        "encoder_slugs": ENCODER_SLUGS,
        "combiner": "logit_average_geometric_mean_of_odds",
        "combiner_weights": weights or {sub: round(1.0 / len(SUBDIRS), 6) for sub in SUBDIRS},
        "weights_source": "single_val_split_fwls" if weights else "fixed_equal_NO_OOF",
        "labeling_from": labeling_from,
        "calibrator_pattern": "streamed_flush_v1+perbc_cal / default_calibrator=identity (unchanged per family)",
        "families": [
            {k: v for k, v in s.items() if k in
             ("subdir", "slug", "code_kind", "runtime_architecture",
              "default_calibrator", "encoder_model_id", "artifacts", "extras")}
            for s in staged
        ],
    }
    (stage / "ensemble_meta.json").write_text(json.dumps(ens_meta, indent=2), encoding="utf-8")

    # 7. Static whitelist import check
    violations = check_imports_whitelist(stage)
    if violations and strict:
        for v in violations:
            LOG.error("IMPORT VIOLATION: %s", v)
        raise RuntimeError(
            f"{len(violations)} whitelist import violation(s); refusing to ship "
            f"(pass strict=False to override). First: {violations[0]}"
        )

    # 8. Calibrator sanity (advisory): all three should be identity default
    cal_warn = []
    for s in staged:
        dc = s.get("default_calibrator")
        if not (isinstance(dc, dict) and dc.get("kind") == "identity"):
            cal_warn.append(f"{s['subdir']}: default_calibrator={dc!r} (expected identity)")
    arch_warn = []
    for s in staged:
        ra = s.get("runtime_architecture")
        if ra and "streamed_flush" not in str(ra):
            arch_warn.append(f"{s['subdir']}: runtime_architecture={ra!r} (expected streamed_flush_v1+perbc_cal)")

    # 9. Manifest of files
    manifest_files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())

    # 10. Zip + size audit
    out = _zip_dir(stage, Path(out_zip))
    size_audit = audit_zip_size(out)
    if not size_audit["ok"] and strict:
        raise RuntimeError(
            f"ZIP {size_audit['size_mb']} MB exceeds {ZIP_LIMIT_MB} MB limit"
        )

    manifest = {
        "out_zip": str(out),
        "stage_dir": str(stage),
        "files": manifest_files,
        "import_violations": violations,
        "calibrator_warnings": cal_warn,
        "arch_warnings": arch_warn,
        "size_audit": size_audit,
        "weights": ens_meta["combiner_weights"],
        "weights_source": ens_meta["weights_source"],
        "families": ens_meta["families"],
    }
    return manifest


# ===========================================================================
# CLI
# ===========================================================================
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--qwen", required=True, help="qwen8b bundle (.zip or dir)")
    ap.add_argument("--nemo", required=True, help="nemotron_trc_v5 bundle (.zip or dir)")
    ap.add_argument("--lgai", required=True, help="LGAI_fixed bundle (.zip or dir)")
    ap.add_argument("--out", required=True, help="output submission .zip")
    ap.add_argument("--val-preds", default=None,
                    help="OPTIONAL .npz/.json of per-model val probs + 'y' for "
                         "single-split FWLS weight fit (NO OOF). Omit => equal weights.")
    ap.add_argument("--labeling-from", default="lgai",
                    choices=["qwen", "nemo", "lgai"],
                    help="which input bundle's labeling.py to ship (default lgai, newest)")
    ap.add_argument("--workspace", default=None, help="staging dir (default: tempdir)")
    ap.add_argument("--no-strict", action="store_true",
                    help="warn instead of failing on import violations / oversize")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # map --labeling-from key (qwen/nemo/lgai) onto subdir-ish key used internally
    manifest = assemble(
        qwen=args.qwen, nemo=args.nemo, lgai=args.lgai, out_zip=args.out,
        val_preds=args.val_preds, labeling_from=args.labeling_from,
        workspace=args.workspace, strict=not args.no_strict,
    )
    print("=" * 70)
    print("ASSEMBLED:", manifest["out_zip"])
    sa = manifest["size_audit"]
    print(f"SIZE: {sa['size_mb']} MB / {sa['limit_mb']} MB  -> {'OK' if sa['ok'] else 'OVER LIMIT'}")
    print("WEIGHTS:", manifest["weights"], f"({manifest['weights_source']})")
    print("IMPORT VIOLATIONS:", manifest["import_violations"] or "none")
    if manifest["calibrator_warnings"]:
        print("CALIBRATOR WARNINGS:", manifest["calibrator_warnings"])
    if manifest["arch_warnings"]:
        print("ARCH WARNINGS:", manifest["arch_warnings"])
    print("FILES:")
    for f in manifest["files"]:
        print("  ", f)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
