"""CSR-backed passrate lookup — the production counterpart of ``DensePassrate``.

On Colab the (subject × item) pass-label matrix is 906 × 311k and sparse; ``src.nn_features
.build_passrate_table`` yields a scipy CSR of mean labels + an observation mask. ``CsrPassrate``
wraps that triple and exposes the SAME interface the codecs use — ``gather(subject, item_keys)``,
``pooled_mean(item_keys, default)``, ``global_mean()`` — so every OOF/leakage guarantee that
was proven against ``DensePassrate`` carries over unchanged.

Stored as the CSR triple ``(indptr, indices, data)`` over subjects × items plus per-column
observed sum/count (for O(len item_keys) pooled difficulty). Pure numpy at query time —
scipy is needed only to *build* the matrix, never to read it — so it is locally testable via
``from_dense``.
"""
from __future__ import annotations

import numpy as np


class CsrPassrate:
    def __init__(self, subject_keys, item_keys, indptr, indices, data):
        self.s_idx = {str(s): i for i, s in enumerate(subject_keys)}
        self.i_idx = {str(it): j for j, it in enumerate(item_keys)}
        self.n_items = len(item_keys)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.float64)
        # per-item (column) observed sum + count → pooled difficulty in O(len cols)
        self._col_sum = np.zeros(self.n_items, dtype=np.float64)
        self._col_cnt = np.zeros(self.n_items, dtype=np.float64)
        np.add.at(self._col_sum, self.indices, self.data)
        np.add.at(self._col_cnt, self.indices, 1.0)
        self._tot_sum = float(self.data.sum())
        self._tot_cnt = float(self.data.size)
        # sorted (subject*n_items + item_col) keys for vectorized pair gather
        subj_nz = np.repeat(np.arange(len(self.s_idx)), np.diff(self.indptr))
        self._pair_keys = subj_nz.astype(np.int64) * self.n_items + self.indices
        order = np.argsort(self._pair_keys, kind="stable")
        self._pair_keys = self._pair_keys[order]
        self._pair_vals = self.data[order]

    # --- constructors ----------------------------------------------------------------
    @classmethod
    def from_dense(cls, subject_keys, item_keys, L):
        """Build from a dense ``L[s, i]`` with ``nan`` = unobserved (tests / small data)."""
        L = np.asarray(L, dtype=float)
        indptr = [0]
        indices, data = [], []
        for s in range(L.shape[0]):
            cols = np.where(np.isfinite(L[s]))[0]
            indices.extend(cols.tolist())
            data.extend(L[s, cols].tolist())
            indptr.append(len(indices))
        return cls(subject_keys, item_keys, indptr, indices, data)

    @classmethod
    def empty(cls, subject_keys, item_keys):
        """A passrate with zero observations (every ``gather`` is nan). Used for the
        fold=all geometry pass, where labels are not read but a passrate is required by
        the codec signature. Builds no dense matrix (safe for 906×311k)."""
        return cls(subject_keys, item_keys, [0] * (len(subject_keys) + 1), [], [])

    @classmethod
    def from_scipy(cls, subject_keys, item_keys, passrate_csr, mask_csr=None):
        """Build from ``src.nn_features.build_passrate_table`` output (Colab).

        ``passrate_csr`` holds the mean labels; its stored nonzeros are the observed cells
        (the parallel ``mask_csr`` is accepted for API symmetry but the CSR's own sparsity
        already encodes observation, matching ``build_passrate_table``)."""
        csr = passrate_csr.tocsr()
        return cls(subject_keys, item_keys, csr.indptr, csr.indices, csr.data)

    # --- read API (matches DensePassrate) --------------------------------------------
    def _row(self, subject):
        si = self.s_idx.get(str(subject))
        if si is None:
            return None, None
        a, b = self.indptr[si], self.indptr[si + 1]
        return self.indices[a:b], self.data[a:b]

    def gather(self, subject, item_keys) -> np.ndarray:
        out = np.full(len(item_keys), np.nan)
        cols, vals = self._row(subject)
        if cols is None:
            return out
        lookup = dict(zip(cols.tolist(), vals.tolist()))  # observed cols → value
        for t, it in enumerate(item_keys):
            j = self.i_idx.get(str(it))
            if j is not None and j in lookup:
                out[t] = lookup[j]
        return out

    def pooled_mean(self, item_keys, default: float = 0.0) -> float:
        cols = [self.i_idx[str(it)] for it in item_keys if str(it) in self.i_idx]
        if not cols:
            return default
        cols = np.asarray(cols, dtype=np.int64)
        cnt = float(self._col_cnt[cols].sum())
        return float(self._col_sum[cols].sum() / cnt) if cnt > 0 else default

    def global_mean(self) -> float:
        return self._tot_sum / self._tot_cnt if self._tot_cnt > 0 else 0.0

    def gather_pairs(self, subject_rows, item_cols) -> np.ndarray:
        """Vectorized parallel-array gather: value at (subject_rows[i], item_cols[i]) or
        ``nan`` if unobserved. The kernel behind a vectorized NN label aggregation — replaces
        the per-row Python ``gather`` loop. O((n log nnz)) via searchsorted on packed keys."""
        sr = np.asarray(subject_rows, dtype=np.int64)
        ic = np.asarray(item_cols, dtype=np.int64)
        out = np.full(sr.shape, np.nan)
        valid = (sr >= 0) & (ic >= 0)
        qk = sr * self.n_items + ic
        pos = np.searchsorted(self._pair_keys, qk)
        pos_c = np.clip(pos, 0, max(self._pair_keys.size - 1, 0))
        hit = valid & (self._pair_keys.size > 0) & (self._pair_keys[pos_c] == qk)
        out[hit] = self._pair_vals[pos_c[hit]]
        return out

    def cluster_aggregates(self, item_to_cluster, n_clusters):
        """Vectorized per-cluster label aggregates in O(nnz) — the fast path that replaces
        ``derive_cluster``'s O(rows × K × members) loop.

        ``item_to_cluster``: int array of length ``n_items`` giving each item column's
        cluster id. Returns ``(difficulty[K], subj_cluster_mean[S,K], subj_cluster_cnt[S,K])``:
          * ``difficulty[k]`` = pooled mean over ALL observed cells whose item is in cluster
            k (empty cluster → global mean). Since the CSR is built from fold-TRAIN rows,
            this is the train-only cluster difficulty — and because the query item is OOF
            (not in train), it is automatically self-excluded.
          * ``subj_cluster_mean[s,k]`` / ``subj_cluster_cnt[s,k]`` = subject s's mean / count
            of observed labels on cluster-k items.
        """
        item_to_cluster = np.asarray(item_to_cluster, dtype=np.int64)
        n_subj = len(self.s_idx)
        cl_nz = item_to_cluster[self.indices]                 # cluster per nonzero
        subj_nz = np.repeat(np.arange(n_subj), np.diff(self.indptr))  # subject per nonzero
        csum = np.zeros(n_clusters); ccnt = np.zeros(n_clusters)
        np.add.at(csum, cl_nz, self.data)
        np.add.at(ccnt, cl_nz, 1.0)
        gm = self.global_mean()
        difficulty = np.where(ccnt > 0, csum / np.maximum(ccnt, 1.0), gm)
        ssum = np.zeros((n_subj, n_clusters)); scnt = np.zeros((n_subj, n_clusters))
        np.add.at(ssum, (subj_nz, cl_nz), self.data)
        np.add.at(scnt, (subj_nz, cl_nz), 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            smean = np.where(scnt > 0, ssum / np.maximum(scnt, 1.0), np.nan)
        return difficulty, smean, scnt
