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
