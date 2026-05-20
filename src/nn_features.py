"""Nearest-neighbor (NN) features for the residual MLP.

This module builds and queries a *full-fidelity* nearest-neighbor index over
training items at training time, and produces an 8-scalar feature vector per
``(subject, item)`` query summarizing the subject's performance on the
top-K nearest training items.

Feature schema (locked, fed into the residual MLP):

    nn_feats[0] = passrate_mean             # mean label of subject on top-K neighbors
    nn_feats[1] = passrate_weighted_mean    # similarity-weighted mean
    nn_feats[2] = passrate_std              # uncertainty signal
    nn_feats[3] = coverage                  # fraction of top-K neighbors with a
                                            # label for this subject
    nn_feats[4] = top1_label                # nearest neighbor's label
                                            # (or top1_missing_sentinel if missing)
    nn_feats[5] = top1_similarity           # similarity of the nearest neighbor
    nn_feats[6] = mean_similarity           # average similarity across top-K
    nn_feats[7] = n_labeled_neighbors_log1p # raw count, log1p-scaled

The pure aggregation helper ``_aggregate_nn_features`` is bit-identical
to the runtime implementation shipped inside ``submission/model.py`` -- the
runtime literally inlines the same function so train and test never drift.
The two-cache design (full-fidelity training index vs. compressed runtime
index) is intentional: training learns on the highest-quality signal, runtime
ships an aggressively PCA'd + int8 approximation, and the residual MLP
absorbs the compression noise via the ``coverage`` / ``n_labeled_neighbors``
features.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

LOG = logging.getLogger("nn_features")

NN_FEATURE_DIM: int = 8
NN_FEATURE_NAMES: tuple[str, ...] = (
    "passrate_mean",
    "passrate_weighted_mean",
    "passrate_std",
    "coverage",
    "top1_label",
    "top1_similarity",
    "mean_similarity",
    "n_labeled_neighbors_log1p",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NNFeaturesConfig:
    """Hyperparameters for nearest-neighbor feature computation.

    ``feature_dim`` is locked to the schema above; it is exposed for
    introspection but not user-tunable.
    """

    enabled: bool = True
    k: int = 16
    similarity: str = "cosine"            # cosine | ip | l2
    feature_dim: int = NN_FEATURE_DIM
    fallback_value: float = 0.0
    top1_missing_sentinel: float = -1.0
    exclude_self_in_training: bool = True
    cache_dir: str = "artifacts/nn_features"

    @classmethod
    def from_dict(cls, d: Mapping | None) -> "NNFeaturesConfig":
        d = dict(d or {})
        return cls(
            enabled=bool(d.get("enabled", True)),
            k=int(d.get("k", 16)),
            similarity=str(d.get("similarity", "cosine")),
            feature_dim=int(d.get("feature_dim", NN_FEATURE_DIM)),
            fallback_value=float(d.get("fallback_value", 0.0)),
            top1_missing_sentinel=float(d.get("top1_missing_sentinel", -1.0)),
            exclude_self_in_training=bool(
                d.get("exclude_self_in_training", True)
            ),
            cache_dir=str(d.get("cache_dir", "artifacts/nn_features")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Numerics: shared aggregation helper (copied verbatim into submission/model.py)
# ---------------------------------------------------------------------------


def _aggregate_nn_features(
    neighbor_passrates: np.ndarray,    # [B, K] mean labels (NaN where missing)
    neighbor_masks: np.ndarray,        # [B, K] 1 where labeled, 0 otherwise
    similarities: np.ndarray,          # [B, K]
    *,
    fallback_value: float,
    top1_missing_sentinel: float,
) -> np.ndarray:
    """Pure NN aggregation. Returns [B, NN_FEATURE_DIM] float32.

    ``neighbor_passrates`` may contain NaN where no observation exists -- the
    mask must independently report which entries are valid. Callers are
    responsible for keeping the two arrays in sync.

    This function is shipped *both* in this training module and inlined
    inside the runtime ``submission/model.py`` (rendered by
    ``export_submission.py``). If you change one, change the other.
    """
    passrates = np.asarray(neighbor_passrates, dtype=np.float32)
    masks = np.asarray(neighbor_masks, dtype=np.float32)
    sims = np.asarray(similarities, dtype=np.float32)

    if passrates.ndim == 1:
        passrates = passrates[None, :]
        masks = masks[None, :]
        sims = sims[None, :]

    B, K = passrates.shape

    # Replace NaNs in passrate with 0 so they don't poison the weighted sums;
    # the mask is the source of truth for "this entry is real".
    pr_safe = np.where(masks > 0, passrates, 0.0).astype(np.float32)

    n_labeled = masks.sum(axis=1)                                # [B]
    has_any = n_labeled > 0

    mean_sim = sims.mean(axis=1).astype(np.float32)              # [B]

    # mean
    pr_sum = pr_safe.sum(axis=1)
    pr_mean = np.where(has_any, pr_sum / np.maximum(n_labeled, 1.0), fallback_value)

    # weighted mean (by similarity, only over labeled entries)
    sim_safe = np.where(masks > 0, sims, 0.0).astype(np.float32)
    # Shift sims into a non-negative weight space. For cosine / IP sims this
    # rescales the [-1, 1] (or unbounded ip) range to [0, +inf); for l2-like
    # distances this would need negation upstream. Documented as "similarity".
    weights = np.clip(sim_safe, 0.0, None)
    weight_sum = weights.sum(axis=1)
    weighted = np.where(
        (weight_sum > 1e-9) & has_any,
        (weights * pr_safe).sum(axis=1) / np.maximum(weight_sum, 1e-9),
        np.where(has_any, pr_mean, fallback_value),
    ).astype(np.float32)

    # std (over labeled entries only; biased / population variance for stability)
    diff = (pr_safe - pr_mean[:, None]) * masks
    sq = (diff * diff).sum(axis=1)
    var = np.where(has_any, sq / np.maximum(n_labeled, 1.0), 0.0)
    pr_std = np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)
    pr_std = np.where(has_any, pr_std, fallback_value).astype(np.float32)

    coverage = (n_labeled / float(max(1, K))).astype(np.float32)

    # top-1: nearest neighbor is column 0 (callers MUST present neighbors in
    # descending similarity order).
    top1_mask = masks[:, 0]
    top1_label = np.where(top1_mask > 0, passrates[:, 0], top1_missing_sentinel)
    top1_label = top1_label.astype(np.float32)
    top1_sim = sims[:, 0].astype(np.float32)

    n_labeled_log = np.log1p(n_labeled).astype(np.float32)

    out = np.stack(
        [
            pr_mean.astype(np.float32),
            weighted,
            pr_std,
            coverage,
            top1_label,
            top1_sim,
            mean_sim,
            n_labeled_log,
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=fallback_value, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# TrainingNNIndex: full-fidelity index used at training / eval time
# ---------------------------------------------------------------------------


class TrainingNNIndex:
    """Full-fidelity NN index over training items.

    Holds ``item_keys`` (list[str]), an fp32 embedding matrix [N, D]
    (optionally L2-normalized when ``similarity == "cosine"``), and an
    optional FAISS index. Persists to ``out_dir`` with deterministic file
    names; the on-disk layout is idempotent (re-using a fresh
    ``build_from_lookup`` call returns the cached files if the item_keys
    match).

    The index never persists self-exclusion state. The caller passes
    ``query_item_keys`` to ``nearest()`` so each query can skip its own row
    when needed.
    """

    EMBEDDINGS_FILE = "training_index_embeddings.npy"
    KEYS_FILE = "training_index_keys.json"
    META_FILE = "training_index_meta.json"
    FAISS_FILE = "training_index.faiss"

    def __init__(self, cfg: NNFeaturesConfig):
        self.cfg = cfg
        self.item_keys: list[str] = []
        self.key_to_row: dict[str, int] = {}
        self.embeddings: np.ndarray | None = None  # [N, D] fp32 (normed for cosine)
        self._faiss_index = None
        self._faiss_attempted = False
        self._faiss_error: str | None = None

    # ------------------------------------------------------------------ build

    @classmethod
    def build_from_lookup(
        cls,
        item_emb_lookup: Mapping[str, np.ndarray],
        out_dir: Path,
        cfg: NNFeaturesConfig,
        *,
        item_keys: list[str] | None = None,
    ) -> "TrainingNNIndex":
        """Stack the embeddings, normalize if cosine, build FAISS, persist.

        Idempotent: if ``out_dir`` already contains a matching index (same
        item_keys, same similarity), the cached files are reused.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if item_keys is None:
            item_keys = sorted(item_emb_lookup.keys())
        else:
            item_keys = [str(k) for k in item_keys]

        # Try to load an existing index whose keys + similarity match.
        existing = cls._maybe_load_existing(out_dir, cfg, item_keys)
        if existing is not None:
            LOG.info(
                "TrainingNNIndex: reusing on-disk index at %s (N=%d D=%d)",
                out_dir,
                existing.embeddings.shape[0],
                existing.embeddings.shape[1],
            )
            return existing

        first = np.asarray(item_emb_lookup[item_keys[0]], dtype=np.float32)
        D = int(first.shape[-1])
        N = len(item_keys)
        emb = np.empty((N, D), dtype=np.float32)
        for i, k in enumerate(item_keys):
            v = np.asarray(item_emb_lookup[k], dtype=np.float32)
            if v.shape != (D,):
                raise ValueError(
                    f"embedding for item_key={k!r} has shape {v.shape}; expected {(D,)}"
                )
            emb[i] = v
        if cfg.similarity == "cosine":
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            emb = (emb / norms).astype(np.float32)

        np.save(out_dir / cls.EMBEDDINGS_FILE, emb)
        (out_dir / cls.KEYS_FILE).write_text(
            json.dumps(item_keys), encoding="utf-8"
        )
        (out_dir / cls.META_FILE).write_text(
            json.dumps(
                {
                    "n_items": N,
                    "dim": D,
                    "similarity": cfg.similarity,
                    "feature_dim": cfg.feature_dim,
                    "k": cfg.k,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        self = cls(cfg)
        self.item_keys = item_keys
        self.key_to_row = {k: i for i, k in enumerate(item_keys)}
        self.embeddings = emb
        self._try_build_faiss(out_dir)
        return self

    @classmethod
    def load(cls, in_dir: Path, cfg: NNFeaturesConfig) -> "TrainingNNIndex":
        in_dir = Path(in_dir)
        item_keys = json.loads((in_dir / cls.KEYS_FILE).read_text(encoding="utf-8"))
        emb = np.load(in_dir / cls.EMBEDDINGS_FILE).astype(np.float32, copy=False)
        self = cls(cfg)
        self.item_keys = list(map(str, item_keys))
        self.key_to_row = {k: i for i, k in enumerate(self.item_keys)}
        self.embeddings = emb
        self._maybe_load_faiss(in_dir)
        return self

    @classmethod
    def _maybe_load_existing(
        cls,
        out_dir: Path,
        cfg: NNFeaturesConfig,
        item_keys: list[str],
    ) -> "TrainingNNIndex | None":
        keys_path = out_dir / cls.KEYS_FILE
        emb_path = out_dir / cls.EMBEDDINGS_FILE
        meta_path = out_dir / cls.META_FILE
        if not (keys_path.exists() and emb_path.exists() and meta_path.exists()):
            return None
        try:
            existing_keys = json.loads(keys_path.read_text(encoding="utf-8"))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if list(map(str, existing_keys)) != list(map(str, item_keys)):
            return None
        if str(meta.get("similarity", "")) != cfg.similarity:
            return None
        return cls.load(out_dir, cfg)

    @classmethod
    def try_load_existing(
        cls,
        out_dir: Path,
        cfg: NNFeaturesConfig,
        item_keys: Sequence[str],
    ) -> "TrainingNNIndex | None":
        """Public cache-only loader. Returns ``None`` if no matching index.

        Use this when the caller already has cached NN feature matrices on
        disk and only needs the index for downstream sanity checks. Unlike
        :meth:`build_from_lookup`, this never falls through to a rebuild --
        it simply returns ``None`` if anything about the cached files does
        not match (keys, similarity, missing files).
        """
        return cls._maybe_load_existing(
            Path(out_dir), cfg, [str(k) for k in item_keys]
        )

    # --------------------------------------------------------------- backend

    def _try_build_faiss(self, out_dir: Path) -> None:
        if self.embeddings is None:
            return
        try:
            import faiss  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._faiss_error = (
                f"faiss not installed ({type(exc).__name__}: {exc})"
            )
            self._faiss_attempted = True
            return
        try:
            N, D = self.embeddings.shape
            if self.cfg.similarity in ("cosine", "ip"):
                index = faiss.IndexFlatIP(D)
            else:
                index = faiss.IndexFlatL2(D)
            index.add(np.ascontiguousarray(self.embeddings, dtype=np.float32))
            faiss.write_index(index, str(out_dir / self.FAISS_FILE))
            self._faiss_index = index
        except Exception as exc:  # noqa: BLE001
            self._faiss_error = f"faiss build failed: {exc}"
        finally:
            self._faiss_attempted = True

    def _maybe_load_faiss(self, in_dir: Path) -> None:
        path = in_dir / self.FAISS_FILE
        if not path.exists():
            return
        try:
            import faiss  # type: ignore

            self._faiss_index = faiss.read_index(str(path))
        except Exception as exc:  # noqa: BLE001
            self._faiss_error = f"faiss read failed: {exc}"

    # --------------------------------------------------------------- queries

    def _normalize_queries(self, queries: np.ndarray) -> np.ndarray:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if self.cfg.similarity == "cosine":
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            q = (q / norms).astype(np.float32)
        return np.ascontiguousarray(q, dtype=np.float32)

    def nearest(
        self,
        query_embeds: np.ndarray,
        k: int | None = None,
        *,
        exclude_self: bool = True,
        query_keys: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (neighbor_indices [B, k], similarities [B, k]).

        When ``exclude_self`` is True and ``query_keys`` matches a training
        item, the first match equal to that key is dropped. Implementation
        uses k+1 candidates and removes the self-match (when present), which
        is robust to both exact-self matches and PCA / quantization-induced
        ordering quirks.
        """
        if self.embeddings is None:
            raise RuntimeError("TrainingNNIndex.nearest called before build")
        k_eff = int(k or self.cfg.k)
        kk = max(1, k_eff)
        # Query k+1 so we can drop the self-match cleanly when present.
        kq = kk + 1
        kq = min(kq, self.embeddings.shape[0])
        q = self._normalize_queries(query_embeds)

        if self._faiss_index is not None:
            try:
                sims, idx = self._faiss_index.search(q, kq)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("FAISS search failed (%s); brute-force fallback", exc)
                sims, idx = self._brute_force(q, kq)
        else:
            sims, idx = self._brute_force(q, kq)

        if self.cfg.similarity == "l2":
            # IndexFlatL2 returns squared distances; flip sign so callers
            # can still treat the second array as "higher = closer".
            sims = -sims

        if exclude_self and query_keys is not None:
            self_rows = np.array(
                [self.key_to_row.get(str(k), -1) for k in query_keys],
                dtype=np.int64,
            )
            idx, sims = self._strip_self(idx, sims, self_rows, kk)
        else:
            idx = idx[:, :kk]
            sims = sims[:, :kk]
        return idx.astype(np.int64, copy=False), sims.astype(np.float32, copy=False)

    def _brute_force(
        self, q: np.ndarray, kq: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # Higher = better for cosine / IP; for L2 we return *squared distances*
        # to match the FAISS convention.
        embs = self.embeddings
        if self.cfg.similarity == "l2":
            d2 = (
                (q * q).sum(axis=1, keepdims=True)
                + (embs * embs).sum(axis=1)[None, :]
                - 2.0 * q @ embs.T
            )
            d2 = np.clip(d2, 0.0, None).astype(np.float32)
            order = np.argpartition(d2, min(kq, d2.shape[1] - 1), axis=1)[:, :kq]
            row_idx = np.arange(d2.shape[0])[:, None]
            best = order[
                row_idx, np.argsort(d2[row_idx, order], axis=1, kind="stable")
            ]
            sims = d2[row_idx, best].astype(np.float32)
            return sims, best.astype(np.int64)
        sims_all = (q @ embs.T).astype(np.float32)
        order = np.argpartition(-sims_all, min(kq, sims_all.shape[1] - 1), axis=1)[
            :, :kq
        ]
        row_idx = np.arange(sims_all.shape[0])[:, None]
        best = order[
            row_idx, np.argsort(-sims_all[row_idx, order], axis=1, kind="stable")
        ]
        sims = sims_all[row_idx, best].astype(np.float32)
        return sims, best.astype(np.int64)

    @staticmethod
    def _strip_self(
        idx: np.ndarray,
        sims: np.ndarray,
        self_rows: np.ndarray,
        kk: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        out_idx = np.empty((idx.shape[0], kk), dtype=np.int64)
        out_sims = np.empty((idx.shape[0], kk), dtype=np.float32)
        for i in range(idx.shape[0]):
            row = idx[i]
            srow = sims[i]
            self_id = int(self_rows[i])
            if self_id < 0:
                out_idx[i] = row[:kk]
                out_sims[i] = srow[:kk]
                continue
            mask = row != self_id
            kept_idx = row[mask][:kk]
            kept_sim = srow[mask][:kk]
            if kept_idx.shape[0] < kk:
                # Edge case: the candidate set was too small even after the
                # +1 buffer. Pad with the most-similar non-self again.
                pad = kk - kept_idx.shape[0]
                kept_idx = np.concatenate([kept_idx, kept_idx[:1].repeat(pad)])
                kept_sim = np.concatenate([kept_sim, kept_sim[:1].repeat(pad)])
            out_idx[i] = kept_idx
            out_sims[i] = kept_sim
        return out_idx, out_sims


# ---------------------------------------------------------------------------
# Sparse pass-rate tables: per-(subject, item) mean label + observation mask
# ---------------------------------------------------------------------------


def build_passrate_table(
    train_df: pd.DataFrame,
    item_index_map: Mapping[str, int],
    subject_index_map: Mapping[str, int],
):
    """Build the sparse pass-rate and observation-mask matrices.

    Returns ``(passrate_csr, mask_csr)`` of shape ``[n_subjects, n_items]``.
    Entries store the mean label of ``(subject, item)`` pairs in training; the
    mask is 1 where there is at least one observation.

    Subject ids that lie outside ``subject_index_map``'s value range are
    skipped (they correspond to UNK at index 0; the caller is responsible
    for setting up UNK semantics).
    """
    from scipy import sparse  # type: ignore

    required = {"subject_key", "item_key", "label"}
    if not required.issubset(train_df.columns):
        raise ValueError(
            f"train_df missing required cols: {sorted(required - set(train_df.columns))}"
        )

    df = train_df[["subject_key", "item_key", "label"]].copy()
    df["subject_key"] = df["subject_key"].astype(str)
    df["item_key"] = df["item_key"].astype(str)
    df = df[df["item_key"].isin(item_index_map)]
    df = df[df["subject_key"].isin(subject_index_map)]
    if df.empty:
        n_subjects = max(int(max(subject_index_map.values()) + 1), 1)
        n_items = max(int(max(item_index_map.values()) + 1), 1)
        empty = sparse.csr_matrix((n_subjects, n_items), dtype=np.float32)
        return empty, empty.copy()

    grouped = (
        df.groupby(["subject_key", "item_key"], sort=False)["label"]
        .mean()
        .reset_index()
    )
    rows = grouped["subject_key"].map(subject_index_map).to_numpy(dtype=np.int64)
    cols = grouped["item_key"].map(item_index_map).to_numpy(dtype=np.int64)
    vals = grouped["label"].astype(np.float32).to_numpy()

    n_subjects = int(max(int(max(subject_index_map.values())) + 1, 1))
    n_items = int(max(int(max(item_index_map.values())) + 1, 1))
    passrate = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_subjects, n_items), dtype=np.float32
    )
    mask = sparse.csr_matrix(
        (np.ones_like(vals, dtype=np.float32), (rows, cols)),
        shape=(n_subjects, n_items),
        dtype=np.float32,
    )
    return passrate, mask


# ---------------------------------------------------------------------------
# Compute NN features for a batch of (subject, item) queries
# ---------------------------------------------------------------------------


def _lookup_neighbor_passrates(
    subject_ids: np.ndarray,         # [B]
    neighbor_indices: np.ndarray,    # [B, K]
    passrate_csr,
    passrate_mask_csr,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse fancy-indexing of (subject, item) -> (passrate, mask).

    Implemented row-by-row on the sparse matrices to keep the working set
    in CSR form (~B small slices); avoids materializing the full
    [n_subjects, n_items] dense matrix.
    """
    B, K = neighbor_indices.shape
    out_pr = np.zeros((B, K), dtype=np.float32)
    out_mask = np.zeros((B, K), dtype=np.float32)

    n_rows = passrate_csr.shape[0]
    pr_indptr = passrate_csr.indptr
    pr_indices = passrate_csr.indices
    pr_data = passrate_csr.data
    mk_indptr = passrate_mask_csr.indptr
    mk_indices = passrate_mask_csr.indices

    # Group by subject id so we touch each CSR row at most once per group.
    for s in np.unique(subject_ids):
        if s < 0 or s >= n_rows:
            continue
        rows_for_subject = np.where(subject_ids == s)[0]
        # Items asked about for this subject (flatten over the group).
        cols_for_subject = neighbor_indices[rows_for_subject]    # [b_s, K]
        # Dense view of the CSR row for this subject; cheap on a single row
        # because csr_matrix exposes the row's indices / data directly.
        start = pr_indptr[s]
        end = pr_indptr[s + 1]
        row_cols = pr_indices[start:end]
        row_vals = pr_data[start:end]
        if row_cols.size:
            # Build a sorted lookup once per subject.
            order = np.argsort(row_cols)
            sorted_cols = row_cols[order]
            sorted_vals = row_vals[order]
            pos = np.searchsorted(sorted_cols, cols_for_subject)
            pos_clipped = np.clip(pos, 0, sorted_cols.size - 1)
            hit = (pos < sorted_cols.size) & (sorted_cols[pos_clipped] == cols_for_subject)
            out_pr[rows_for_subject] = np.where(hit, sorted_vals[pos_clipped], 0.0)
        # mask
        mstart = mk_indptr[s]
        mend = mk_indptr[s + 1]
        m_cols = mk_indices[mstart:mend]
        if m_cols.size:
            order = np.argsort(m_cols)
            sorted_m = m_cols[order]
            pos = np.searchsorted(sorted_m, cols_for_subject)
            pos_clipped = np.clip(pos, 0, sorted_m.size - 1)
            hit = (pos < sorted_m.size) & (sorted_m[pos_clipped] == cols_for_subject)
            out_mask[rows_for_subject] = hit.astype(np.float32)
    return out_pr, out_mask


def compute_nn_features(
    query_embeds: np.ndarray,
    query_item_keys: list[str] | None,
    subject_ids: np.ndarray,
    nn_index: TrainingNNIndex,
    passrate_csr,
    passrate_mask_csr,
    cfg: NNFeaturesConfig,
    *,
    exclude_self: bool | None = None,
) -> np.ndarray:
    """Return ``[B, NN_FEATURE_DIM]`` float32 NN feature matrix.

    Steps:
    1. Get top-k neighbors via ``nn_index.nearest``.
    2. Look up per-(subject, neighbor) pass-rate + mask from the sparse
       matrices (vectorized via fancy indexing).
    3. Aggregate into the 8-scalar feature vector via
       ``_aggregate_nn_features``.
    """
    if exclude_self is None:
        exclude_self = bool(cfg.exclude_self_in_training)
    k = int(cfg.k)
    neighbor_idx, sims = nn_index.nearest(
        query_embeds,
        k=k,
        exclude_self=exclude_self,
        query_keys=query_item_keys,
    )
    subject_ids = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
    if subject_ids.shape[0] != neighbor_idx.shape[0]:
        raise ValueError(
            f"subject_ids length {subject_ids.shape[0]} != queries {neighbor_idx.shape[0]}"
        )
    passrates, masks = _lookup_neighbor_passrates(
        subject_ids, neighbor_idx, passrate_csr, passrate_mask_csr
    )
    return _aggregate_nn_features(
        passrates,
        masks,
        sims,
        fallback_value=float(cfg.fallback_value),
        top1_missing_sentinel=float(cfg.top1_missing_sentinel),
    )


__all__ = [
    "NN_FEATURE_DIM",
    "NN_FEATURE_NAMES",
    "NNFeaturesConfig",
    "TrainingNNIndex",
    "_aggregate_nn_features",
    "build_passrate_table",
    "compute_nn_features",
]
