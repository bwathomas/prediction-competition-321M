"""Per-row metadata feature builder for ``src.member2_metadata_mlp``.

SHIP_PLAN_3WAY §"UPDATE (MLP recipe CHANGED)": the ship MLPs must be trained the
AIDE way **+ subject + subject-metadata + benchmark + benchmark-metadata**. That
"+metadata" channel is exactly what ``src.member2_metadata_mlp.fit_member2_metadata_mlp``
consumes: 7 categorical id arrays (subject / bc / cluster / family / macro_family /
organization / bench_topic) + a numerical matrix + cardinalities. This module is the
single source of truth for building those arrays, aligned to the canonical ship rows
(``DR/ship/rows/_tr_subj.npy`` etc.).

----------------------------------------------------------------------------------
WHAT fit_member2_metadata_mlp EXPECTS (read off src/member2_metadata_mlp.py)
  Categorical (each [N] int64, 0-based into a per-field vocab; the model adds its own
  trailing UNK slot at index n_<field>, so any out-of-range id is treated as cold-start):
      subject_ids, bc_ids, cluster_ids, family_ids, macro_family_ids,
      organization_ids, bench_topic_ids
  Numerical ([N, n_num] float32), in the LOCKED block order (assemble_numerical):
      subject_numerical | bench_numerical | bc_redacted_flag(1) | marginals
      with n_num == n_subj_num + n_bench_num + 1 + n_marginals.
  Provenance / cardinalities:
      subject_keys (len n_subjects), bc_keys (len n_bcs),
      n_subjects, n_bcs, n_clusters, n_families, n_macro_families, n_organizations,
      n_bench_topics, n_subj_num, n_bench_num, n_marginals, num_feature_names.

  build_metadata_tables(...) returns a dict carrying EXACTLY these keys (plus y is the
  caller's to add), ready to splat:
      st = fit_member2_metadata_mlp(y=y_train, **build_metadata_tables(rows_subj=_tr_subj,
               rows_item=_tr_item, ...))

----------------------------------------------------------------------------------
KEY / FIELD DERIVATION (read off src/data.py + src/metadata_features.py + aide/features/metadata.py)
  * subject_key  = sha256(subject_content)                         [src.data.add_stable_keys]
  * bc_key       = "{benchmark}::{condition}"  (benchmark_condition_key)
  * condition    = normalize_condition(raw)  -> literal "none" sentinel for null/blank
  * subject display name = the "Name: <x>" line of subject_content  [extract_display_name]
  * subject -> {organization, family, macro_family} via JOIN on model_info.csv:
        exact match on `name`, else suffix (after last '/') fallback   [aide.features.metadata]
        model_info.csv columns: name, organization, parameters, release_date, family, macro-family
  * subject numerics: log1p(parameters) [=log_params], release_date  [MetadataSchema.subject_numeric]
        each emitted as (z-scored value, missing-flag) -> 2 cols/field -> n_subj_num = 4
  * benchmark -> {topic, age} via JOIN on benchmark_info.csv (exact on `benchmark`)
        benchmark_info.csv columns: benchmark, topic, age, has_conditions, conditions
  * benchmark numerics: benchmark_age (= the CSV `age`)             [MetadataSchema.benchmark_numeric]
        emitted as (z-scored value, missing-flag) -> 2 cols -> n_bench_num = 2
  * bc_redacted_flag: 1.0 when this row's benchmark is NOT found in benchmark_info.csv
        (i.e. its benchmark-side metadata was unavailable / "redacted"), else 0.0.
  * cluster_ids: item-level kmeans cluster id from DR/artifacts/nn_features/item_clusters.parquet
        joined by item_key. MISSING (no cluster artifact / unmatched item) -> all-zeros
        cluster_ids + n_clusters=1 so the model's single cluster row is a benign constant.
  * marginals: per-row prior columns the metadata MLP folds into the numeric channel.
        DEFAULT here = [subject_passrate, benchmark_passrate] computed from TRAIN labels
        only (leak-free: holdout uses the train-derived means), n_marginals=2. Pass
        marginals=None to disable (n_marginals=0).

----------------------------------------------------------------------------------
DRIVE PATHS IT READS (drive_root DR = /content/drive/MyDrive/prediction-competition-321M)
  * model_info.csv      : DR/data/metadata/model_info.csv      (also vendored in repo data/metadata/)
  * benchmark_info.csv  : DR/data/metadata/benchmark_info.csv  (also vendored in repo)
  * labels / enrichment : DR/prepared_datasets/*measurement_db_prepared*.parquet
        columns used: subject_key, item_key, benchmark, condition, subject_content, label
        (subject_key/benchmark/condition/subject_content are the "+ enrichment" cols the
        SHIP_PLAN notes; subj_emb_* are NOT needed by this metadata path.)
  * canonical rows      : DR/ship/rows/{_tr_subj,_tr_item,_ho_subj,_ho_item}.npy
  * clusters (optional) : DR/artifacts/nn_features/item_clusters.parquet  (item_key -> cluster_id)

MISSING-FIELD FALLBACKS (each one is non-fatal and flagged in the returned 'report'):
  * rows .npy are KEYS by default (per SHIP_PLAN A1: per-row sha256 keys). If they are
    integer INDICES, the caller can resolve them upstream; this module asserts the row
    lengths and that the row strings look like 64-hex sha256 (warns otherwise).
  * model_info / benchmark_info absent -> that side's categorical -> all MISSING token,
    numerics -> missing-flag=1, value=0; cardinality still >=1 (one MISSING row).
  * item_clusters.parquet absent -> n_clusters=1, cluster_ids all 0 (constant; harmless).
  * a subject/benchmark unmatched in the CSV -> MISSING token id, missing-flag=1.

The module imports ONLY numpy + pandas (+ optional pyarrow for parquet). It does NOT
import torch and is safe to call from a training-time Colab cell BEFORE fit.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Constants pinned to the consumer's runtime contract.
# --------------------------------------------------------------------------------------

# Subject display name regex — anchored "Name: <x>" line of subject_content. Mirrors
# src.metadata_features.extract_display_name / src.data._NAME_LINE.
_NAME_RE = re.compile(r"(?im)^\s*Name:\s*(.*?)\s*$")

# Tokens reserved by the field vocabularies. Index 0 == MISSING ("I have no value").
# (The metadata MLP itself adds a TRAILING UNK slot at index n_field for cold-start;
#  our 0-based vocab therefore does NOT reserve a row for UNK — only for MISSING.)
_MISSING = "__MISSING__"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------------------
# Small, dependency-light helpers
# --------------------------------------------------------------------------------------


def _normalize_condition(value: object) -> str:
    """Literal 'none' sentinel for null/blank — mirrors src.data.normalize_condition."""
    if value is None:
        return "none"
    s = str(value)
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "none"
    return s


def _extract_display_name(subject_content: object) -> str:
    if not isinstance(subject_content, str):
        return ""
    m = _NAME_RE.search(subject_content)
    return m.group(1).strip() if m else ""


def _clean_cat(v: object) -> str | None:
    """Return a cleaned categorical token, or None for MISSING."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null", "unknown", "unk"}:
        return None
    return s


class _Vocab:
    """0-based MISSING@0 vocabulary; unseen-at-encode -> MISSING (model adds UNK slot)."""

    def __init__(self) -> None:
        self.tok2id: dict[str, int] = {_MISSING: 0}

    def fit(self, values) -> "_Vocab":
        for v in values:
            k = _clean_cat(v)
            if k is not None and k not in self.tok2id:
                self.tok2id[k] = len(self.tok2id)
        return self

    def encode(self, values) -> np.ndarray:
        out = np.zeros(len(values), dtype=np.int64)
        for i, v in enumerate(values):
            k = _clean_cat(v)
            out[i] = self.tok2id.get(k, 0) if k is not None else 0
        return out

    @property
    def n(self) -> int:
        return len(self.tok2id)


def _zscore_with_missing(
    raw: np.ndarray, train_mask: np.ndarray, log1p: bool
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(z-scored value, missing-flag) using TRAIN-slice mean/std. Leak-free."""
    x = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=np.float64)
    missing = (~np.isfinite(x)).astype(np.float32)
    if log1p:
        x = np.log1p(np.maximum(x, 0.0))
    fin_train = x[train_mask & np.isfinite(x)]
    if fin_train.size:
        mu = float(np.mean(fin_train))
        sd = float(np.std(fin_train)) or 1.0
    else:
        mu, sd = 0.0, 1.0
    filled = np.where(np.isfinite(x), x, mu)
    z = ((filled - mu) / sd).astype(np.float32)
    return z, missing, mu, sd


def _read_csv(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def _read_parquet(path: str | Path, columns: Sequence[str] | None = None):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p, columns=list(columns) if columns else None)
    except Exception:
        # Column subset may not exist; fall back to full read.
        return pd.read_parquet(p)


def _first_glob(pattern: str) -> Path | None:
    import glob

    hits = sorted(glob.glob(pattern))
    return Path(hits[0]) if hits else None


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def build_metadata_tables(
    *,
    # Canonical per-row keys (the ship row order). Each is a 1-D array/list of strings.
    rows_subj: Sequence[str],          # subject_key per row  (DR/ship/rows/_*_subj.npy)
    rows_item: Sequence[str],          # item_key per row     (DR/ship/rows/_*_item.npy)
    # Optional second split appended AFTER the first (e.g. train then holdout). When
    # given, the returned arrays cover the concatenation [rows_subj+rows_subj2, ...] and
    # 'n_first' marks the boundary; vocabs/scalers/marginals are fit on the FIRST split
    # only (leak-free), encoding both.
    rows_subj2: Sequence[str] | None = None,
    rows_item2: Sequence[str] | None = None,
    # Where to read the joins from. Pass the Drive paths on Colab; the repo-vendored
    # copies are sensible local defaults.
    model_info_path: str | Path = "data/metadata/model_info.csv",
    benchmark_info_path: str | Path = "data/metadata/benchmark_info.csv",
    prepared_parquet_path: str | Path | None = None,   # *measurement_db_prepared*.parquet
    prepared_glob: str | None = None,                  # alt: glob to resolve the parquet
    item_clusters_path: str | Path | None = None,      # DR/artifacts/nn_features/item_clusters.parquet
    # Marginal (prior) columns folded into the numeric channel. "passrate" => append
    # [subject_passrate, benchmark_passrate] computed from FIRST-split labels. None => no
    # marginals (n_marginals=0). Or pass a precomputed (N_total, M) float array directly.
    marginals: str | np.ndarray | None = "passrate",
    # If the rows arrays are integer indices instead of keys, resolve them upstream and
    # pass key arrays; this flag only relaxes the sha256 sanity check.
    rows_are_keys: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build the full splat-dict for ``fit_member2_metadata_mlp``.

    Returns a dict with EVERY non-``y`` keyword of ``fit_member2_metadata_mlp`` plus a
    diagnostic ``"report"`` entry (coverage fractions + fallbacks taken). Splat it:

        out = build_metadata_tables(rows_subj=_tr_subj, rows_item=_tr_item, ...)
        report = out.pop("report")
        n_first = out.pop("n_first")          # boundary if a 2nd split was passed
        state = fit_member2_metadata_mlp(y=y_train, **out)

    When ``rows_subj2`` is given, slice the returned arrays at ``n_first`` to recover the
    holdout block for prediction:  arr[n_first:] is the second split, row-aligned.
    """
    report: dict[str, Any] = {"fallbacks": [], "coverage": {}}

    subj1 = [str(s) for s in np.asarray(rows_subj).reshape(-1).tolist()]
    item1 = [str(s) for s in np.asarray(rows_item).reshape(-1).tolist()]
    if len(subj1) != len(item1):
        raise ValueError(
            f"rows_subj ({len(subj1)}) and rows_item ({len(item1)}) length mismatch"
        )
    n_first = len(subj1)

    if rows_subj2 is not None:
        subj2 = [str(s) for s in np.asarray(rows_subj2).reshape(-1).tolist()]
        item2 = [str(s) for s in np.asarray(rows_item2).reshape(-1).tolist()]
        if len(subj2) != len(item2):
            raise ValueError(
                f"rows_subj2 ({len(subj2)}) / rows_item2 ({len(item2)}) length mismatch"
            )
    else:
        subj2, item2 = [], []

    subj_keys = subj1 + subj2
    item_keys = item1 + item2
    N = len(subj_keys)
    train_mask = np.zeros(N, dtype=bool)
    train_mask[:n_first] = True

    if rows_are_keys:
        bad = sum(1 for s in subj_keys[:50] if not _HEX64.match(s))
        if bad and verbose:
            print(
                f"[metadata_tables] WARN: {bad}/50 sampled subject rows are not 64-hex "
                "sha256 keys; if these are integer indices pass rows_are_keys=False and "
                "resolve to keys upstream."
            )
            report["fallbacks"].append("rows_not_sha256_like")

    # ---- Resolve + load the prepared parquet (the per-row enrichment table) ----
    if prepared_parquet_path is None and prepared_glob:
        resolved = _first_glob(prepared_glob)
        prepared_parquet_path = str(resolved) if resolved else None
    prep_cols = [
        "subject_key", "item_key", "benchmark", "condition", "subject_content", "label",
    ]
    prepared = (
        _read_parquet(prepared_parquet_path, columns=prep_cols)
        if prepared_parquet_path
        else None
    )
    if prepared is None:
        raise FileNotFoundError(
            "prepared parquet not found — pass prepared_parquet_path or prepared_glob "
            "(DR/prepared_datasets/*measurement_db_prepared*.parquet). It carries the "
            "benchmark/condition/subject_content enrichment + label keyed by "
            "(subject_key, item_key)."
        )
    for c in ("subject_key", "item_key"):
        prepared[c] = prepared[c].astype(str)

    # Per (subject_key,item_key) -> enrichment row. The prepared table is the union of
    # train+holdout; we look up each canonical row's (subj,item) pair.
    have_cols = set(prepared.columns)
    # subject_content is keyed by subject_key alone (1:1); benchmark/condition by item_key.
    subj_content_by_key: dict[str, str] = {}
    if "subject_content" in have_cols:
        sc = prepared.drop_duplicates("subject_key")
        subj_content_by_key = dict(
            zip(sc["subject_key"].astype(str), sc["subject_content"].astype(str))
        )
    bench_by_item: dict[str, str] = {}
    cond_by_item: dict[str, str] = {}
    if "benchmark" in have_cols:
        bi = prepared.drop_duplicates("item_key")
        bench_by_item = dict(
            zip(bi["item_key"].astype(str), bi["benchmark"].astype(str))
        )
        if "condition" in have_cols:
            cond_by_item = dict(
                zip(bi["item_key"].astype(str), bi["condition"].astype(str))
            )

    # Per-row benchmark / condition / subject display name.
    benchmarks = [bench_by_item.get(k) for k in item_keys]
    conditions = [_normalize_condition(cond_by_item.get(k)) for k in item_keys]
    display_names = [
        _extract_display_name(subj_content_by_key.get(k, "")) for k in subj_keys
    ]
    # bc_key = "{benchmark}::{condition}" (== src.data.benchmark_condition_key).
    bc_row_keys = [
        f"{(b if b is not None else '__MISSING__')}::{c}"
        for b, c in zip(benchmarks, conditions)
    ]

    n_bench_resolved = sum(1 for b in benchmarks if b is not None)
    report["coverage"]["benchmark_in_prepared"] = n_bench_resolved / max(N, 1)
    n_name_resolved = sum(1 for d in display_names if d)
    report["coverage"]["subject_name_in_prepared"] = n_name_resolved / max(N, 1)

    # ---- subject id + bc id vocabularies (fit on FIRST split, encode all) ----
    subj_vocab = _Vocab().fit(subj_keys[:n_first])
    bc_vocab = _Vocab().fit(bc_row_keys[:n_first])
    subject_ids = subj_vocab.encode(subj_keys)
    bc_ids = bc_vocab.encode(bc_row_keys)
    n_subjects = subj_vocab.n
    n_bcs = bc_vocab.n
    # Provenance lists (len == cardinality), in id order. MISSING token at index 0.
    inv_subj = {v: k for k, v in subj_vocab.tok2id.items()}
    inv_bc = {v: k for k, v in bc_vocab.tok2id.items()}
    subject_keys_out = tuple(inv_subj[i] for i in range(n_subjects))
    bc_keys_out = tuple(inv_bc[i] for i in range(n_bcs))

    # ---- model_info.csv join: subject -> organization / family / macro_family + nums ----
    model_info = _read_csv(model_info_path)
    org_row = [None] * N
    fam_row = [None] * N
    macro_row = [None] * N
    params_row = np.full(N, np.nan, dtype=np.float64)
    release_row = np.full(N, np.nan, dtype=np.float64)
    if model_info is None:
        report["fallbacks"].append("model_info_missing")
        if verbose:
            print(f"[metadata_tables] WARN: model_info.csv not at {model_info_path}")
    else:
        mi = model_info.copy()
        mi.columns = [str(c) for c in mi.columns]
        if "macro_family" not in mi.columns and "macro-family" in mi.columns:
            mi = mi.rename(columns={"macro-family": "macro_family"})
        if "name" not in mi.columns:
            for cand in ("model_id", "model", "display_name"):
                if cand in mi.columns:
                    mi = mi.rename(columns={cand: "name"})
                    break
        exact: dict[str, Any] = {}
        suffix: dict[str, Any] = {}
        for _, r in mi.iterrows():
            nm = str(r.get("name", ""))
            exact[nm] = r
            suffix[nm.split("/")[-1]] = r
        hits = 0
        for i, nm in enumerate(display_names):
            r = exact.get(str(nm))
            if r is None:
                r = suffix.get(str(nm).split("/")[-1])
            if r is not None:
                hits += 1
                org_row[i] = r.get("organization")
                fam_row[i] = r.get("family")
                macro_row[i] = r.get("macro_family")
                params_row[i] = pd.to_numeric(r.get("parameters"), errors="coerce")
                release_row[i] = pd.to_numeric(r.get("release_date"), errors="coerce")
        report["coverage"]["subject_meta_join"] = hits / max(N, 1)
        if verbose:
            print(
                f"[metadata_tables] subject->model_info join coverage: "
                f"{hits}/{N} = {hits / max(N, 1):.3f}"
            )

    family_vocab = _Vocab().fit([fam_row[i] for i in range(n_first)])
    macro_vocab = _Vocab().fit([macro_row[i] for i in range(n_first)])
    org_vocab = _Vocab().fit([org_row[i] for i in range(n_first)])
    family_ids = family_vocab.encode(fam_row)
    macro_family_ids = macro_vocab.encode(macro_row)
    organization_ids = org_vocab.encode(org_row)
    n_families = family_vocab.n
    n_macro_families = macro_vocab.n
    n_organizations = org_vocab.n

    # Subject numerics: log_params + release_date -> (value, missing) each => 4 cols.
    lp_z, lp_m, _, _ = _zscore_with_missing(params_row, train_mask, log1p=True)
    rd_z, rd_m, _, _ = _zscore_with_missing(release_row, train_mask, log1p=False)
    subject_numerical = np.stack([lp_z, lp_m, rd_z, rd_m], axis=1).astype(np.float32)
    subj_num_names = ["log_params", "log_params__missing", "release_date", "release_date__missing"]
    n_subj_num = 4

    # ---- benchmark_info.csv join: benchmark -> topic / age ----
    benchmark_info = _read_csv(benchmark_info_path)
    topic_row = [None] * N
    age_row = np.full(N, np.nan, dtype=np.float64)
    bench_known = np.zeros(N, dtype=bool)
    if benchmark_info is None:
        report["fallbacks"].append("benchmark_info_missing")
        if verbose:
            print(
                f"[metadata_tables] WARN: benchmark_info.csv not at {benchmark_info_path}"
            )
    else:
        bench_info = benchmark_info.copy()
        bench_info.columns = [str(c) for c in bench_info.columns]
        if "benchmark" not in bench_info.columns:
            for cand in ("benchmark_id", "name"):
                if cand in bench_info.columns:
                    bench_info = bench_info.rename(columns={cand: "benchmark"})
                    break
        by_b = {str(r["benchmark"]): r for _, r in bench_info.iterrows()}
        bhits = 0
        for i, b in enumerate(benchmarks):
            r = by_b.get(str(b)) if b is not None else None
            if r is not None:
                bhits += 1
                bench_known[i] = True
                topic_row[i] = r.get("topic")
                age_row[i] = pd.to_numeric(r.get("age"), errors="coerce")
        report["coverage"]["benchmark_meta_join"] = bhits / max(N, 1)
        if verbose:
            print(
                f"[metadata_tables] benchmark->benchmark_info join coverage: "
                f"{bhits}/{N} = {bhits / max(N, 1):.3f}"
            )

    topic_vocab = _Vocab().fit([topic_row[i] for i in range(n_first)])
    bench_topic_ids = topic_vocab.encode(topic_row)
    n_bench_topics = topic_vocab.n

    age_z, age_m, _, _ = _zscore_with_missing(age_row, train_mask, log1p=False)
    bench_numerical = np.stack([age_z, age_m], axis=1).astype(np.float32)
    bench_num_names = ["benchmark_age", "benchmark_age__missing"]
    n_bench_num = 2

    # bc_redacted_flag: benchmark metadata unavailable for this row -> 1.0.
    bc_redacted_flag = (~bench_known).astype(np.float32)

    # ---- cluster_ids from the item_clusters artifact (optional) ----
    cluster_ids = np.zeros(N, dtype=np.int64)
    n_clusters = 1
    clu_df = _read_parquet(item_clusters_path) if item_clusters_path else None
    if clu_df is not None:
        cols = {c.lower(): c for c in clu_df.columns}
        ik = cols.get("item_key")
        cc = cols.get("cluster_id") or cols.get("cluster") or cols.get("fine")
        if ik is not None and cc is not None:
            clu_map = dict(
                zip(clu_df[ik].astype(str), pd.to_numeric(clu_df[cc], errors="coerce"))
            )
            raw = np.array(
                [clu_map.get(k, np.nan) for k in item_keys], dtype=np.float64
            )
            # Densify cluster ids to a contiguous 0-based range over observed values.
            seen = sorted({int(v) for v in raw if np.isfinite(v)})
            remap = {c: j for j, c in enumerate(seen)}
            if seen:
                cluster_ids = np.array(
                    [remap[int(v)] if np.isfinite(v) else len(seen) for v in raw],
                    dtype=np.int64,
                )
                # Unmatched items (NaN) -> trailing UNK slot (== n_clusters) handled by model.
                n_clusters = len(seen)
                chits = int(np.isfinite(raw).sum())
                report["coverage"]["item_cluster_join"] = chits / max(N, 1)
                if verbose:
                    print(
                        f"[metadata_tables] item cluster join coverage: "
                        f"{chits}/{N} = {chits / max(N, 1):.3f}  (n_clusters={n_clusters})"
                    )
            else:
                report["fallbacks"].append("item_clusters_empty")
        else:
            report["fallbacks"].append("item_clusters_no_usable_cols")
    else:
        report["fallbacks"].append("item_clusters_missing")
        if verbose:
            print(
                "[metadata_tables] item_clusters.parquet absent -> n_clusters=1, "
                "cluster_ids all 0 (constant; harmless)."
            )

    # ---- marginals (prior columns) ----
    if isinstance(marginals, np.ndarray):
        marg = np.asarray(marginals, dtype=np.float32)
        if marg.ndim != 2 or marg.shape[0] != N:
            raise ValueError(
                f"marginals array must be (N={N}, M); got {marg.shape}"
            )
        marginal_names = [f"marg_{j}" for j in range(marg.shape[1])]
    elif marginals == "passrate":
        # Leak-free passrate priors from the FIRST split's labels only.
        lab = prepared.drop_duplicates(["subject_key", "item_key"]).copy()
        lab["subject_key"] = lab["subject_key"].astype(str)
        lab["item_key"] = lab["item_key"].astype(str)
        lab_map = dict(
            zip(
                zip(lab["subject_key"], lab["item_key"]),
                pd.to_numeric(lab.get("label"), errors="coerce"),
            )
        )
        y_first = np.array(
            [lab_map.get((subj_keys[i], item_keys[i]), np.nan) for i in range(n_first)],
            dtype=np.float64,
        )
        global_mean = float(np.nanmean(y_first)) if np.isfinite(y_first).any() else 0.5
        # subject passrate (train mean per subject id), benchmark passrate (per benchmark).
        subj_sum: dict[int, float] = {}
        subj_cnt: dict[int, int] = {}
        bench_sum: dict[str, float] = {}
        bench_cnt: dict[str, int] = {}
        for i in range(n_first):
            yv = y_first[i]
            if not np.isfinite(yv):
                continue
            sid = int(subject_ids[i])
            subj_sum[sid] = subj_sum.get(sid, 0.0) + yv
            subj_cnt[sid] = subj_cnt.get(sid, 0) + 1
            bk = benchmarks[i] if benchmarks[i] is not None else "__MISSING__"
            bench_sum[bk] = bench_sum.get(bk, 0.0) + yv
            bench_cnt[bk] = bench_cnt.get(bk, 0) + 1
        subj_pr = np.empty(N, dtype=np.float32)
        bench_pr = np.empty(N, dtype=np.float32)
        for i in range(N):
            sid = int(subject_ids[i])
            subj_pr[i] = (
                subj_sum[sid] / subj_cnt[sid]
                if subj_cnt.get(sid, 0) > 0
                else global_mean
            )
            bk = benchmarks[i] if benchmarks[i] is not None else "__MISSING__"
            bench_pr[i] = (
                bench_sum[bk] / bench_cnt[bk]
                if bench_cnt.get(bk, 0) > 0
                else global_mean
            )
        marg = np.stack([subj_pr, bench_pr], axis=1).astype(np.float32)
        marginal_names = ["subject_passrate", "benchmark_passrate"]
    elif marginals is None:
        marg = np.zeros((N, 0), dtype=np.float32)
        marginal_names = []
    else:
        raise ValueError(
            f"marginals must be 'passrate', None, or an (N,M) array; got {marginals!r}"
        )
    n_marginals = int(marg.shape[1])

    # ---- assemble numerical in the LOCKED block order ----
    # subject_numerical | bench_numerical | bc_redacted_flag(1) | marginals
    numerical = np.concatenate(
        [
            subject_numerical,
            bench_numerical,
            bc_redacted_flag.reshape(-1, 1),
            marg,
        ],
        axis=1,
    ).astype(np.float32)
    num_feature_names = (
        [f"subj_num__{n}" for n in subj_num_names]
        + [f"bench_num__{n}" for n in bench_num_names]
        + ["bc_redacted_flag"]
        + [f"marg__{n}" for n in marginal_names]
    )
    n_num = numerical.shape[1]
    assert n_num == n_subj_num + n_bench_num + 1 + n_marginals, (
        f"n_num bookkeeping bug: {n_num} != {n_subj_num}+{n_bench_num}+1+{n_marginals}"
    )
    assert len(num_feature_names) == n_num

    report["cardinalities"] = {
        "n_subjects": n_subjects,
        "n_bcs": n_bcs,
        "n_clusters": n_clusters,
        "n_families": n_families,
        "n_macro_families": n_macro_families,
        "n_organizations": n_organizations,
        "n_bench_topics": n_bench_topics,
    }
    report["n_num"] = n_num
    report["n_rows"] = N
    report["n_first"] = n_first

    out: dict[str, Any] = {
        # categorical id arrays (len N)
        "subject_ids": subject_ids,
        "bc_ids": bc_ids,
        "cluster_ids": cluster_ids,
        "family_ids": family_ids,
        "macro_family_ids": macro_family_ids,
        "organization_ids": organization_ids,
        "bench_topic_ids": bench_topic_ids,
        # numerical
        "numerical": numerical,
        # provenance / cardinalities
        "subject_keys": subject_keys_out,
        "bc_keys": bc_keys_out,
        "num_feature_names": tuple(num_feature_names),
        "n_subjects": n_subjects,
        "n_bcs": n_bcs,
        "n_clusters": n_clusters,
        "n_families": n_families,
        "n_macro_families": n_macro_families,
        "n_organizations": n_organizations,
        "n_bench_topics": n_bench_topics,
        "n_subj_num": n_subj_num,
        "n_bench_num": n_bench_num,
        "n_marginals": n_marginals,
        # caller bookkeeping (pop before splat into fit_member2_metadata_mlp)
        "n_first": n_first,
        "report": report,
    }
    return out


__all__ = ["build_metadata_tables"]


# --------------------------------------------------------------------------------------
# Self-test (runs on the repo-vendored CSVs with a tiny synthetic row set).
# --------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    # Synthetic prepared parquet + rows: 4 train rows, 2 holdout rows.
    def _sha(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()

    subj_contents = {
        "A": "Name: meta-llama/Llama-3-8B\nFamily: Llama 3",
        "B": "Name: mistralai/Mistral-7B\nFamily: Mistral",
    }
    rows = []
    sk = {n: _sha(c) for n, c in subj_contents.items()}
    for subj, bench, cond, lab in [
        ("A", "afrimedqa", "none", 1),
        ("A", "agentdojo", "attack", 0),
        ("B", "afrimedqa", "none", 1),
        ("B", "unknownbench", "none", 0),
        ("A", "afrimedqa", "none", 1),   # holdout
        ("B", "agentdojo", "metric", 0),  # holdout
    ]:
        ik = _sha(f"{bench}|{cond}|q")
        rows.append(
            dict(
                subject_key=sk[subj],
                item_key=ik,
                benchmark=bench,
                condition=cond,
                subject_content=subj_contents[subj],
                label=lab,
            )
        )
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as td:
        pq = Path(td) / "prep.parquet"
        df.to_parquet(pq)
        out = build_metadata_tables(
            rows_subj=[r["subject_key"] for r in rows[:4]],
            rows_item=[r["item_key"] for r in rows[:4]],
            rows_subj2=[r["subject_key"] for r in rows[4:]],
            rows_item2=[r["item_key"] for r in rows[4:]],
            prepared_parquet_path=str(pq),
            model_info_path="data/metadata/model_info.csv",
            benchmark_info_path="data/metadata/benchmark_info.csv",
            verbose=True,
        )
    rep = out.pop("report")
    nf = out.pop("n_first")
    print("\n--- SELF TEST OK ---")
    print("n_first:", nf, " n_rows:", len(out["subject_ids"]))
    print("cardinalities:", rep["cardinalities"])
    print("num_feature_names:", out["num_feature_names"])
    print("n_num:", out["numerical"].shape[1], " block sum check:",
          out["n_subj_num"] + out["n_bench_num"] + 1 + out["n_marginals"])
    print("fallbacks:", rep["fallbacks"])
    print("coverage:", rep["coverage"])
    # Every key fit_member2_metadata_mlp needs must be present.
    need = {
        "subject_ids", "bc_ids", "cluster_ids", "family_ids", "macro_family_ids",
        "organization_ids", "bench_topic_ids", "numerical", "subject_keys", "bc_keys",
        "num_feature_names", "n_subjects", "n_bcs", "n_clusters", "n_families",
        "n_macro_families", "n_organizations", "n_bench_topics", "n_subj_num",
        "n_bench_num", "n_marginals",
    }
    missing = need - set(out)
    assert not missing, f"missing keys for fit_member2_metadata_mlp: {missing}"
    assert len(out["subject_keys"]) == out["n_subjects"]
    assert len(out["bc_keys"]) == out["n_bcs"]
    print("ALL CONTRACT KEYS PRESENT.")
