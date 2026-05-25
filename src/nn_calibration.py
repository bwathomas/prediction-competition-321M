"""Netflix-Prize-style nearest-neighbor calibration.

The trained model produces an *uncalibrated* probability ``p_uncal`` for
every (subject, item) pair. Two systematic miscalibrations show up in
practice on this dataset:

1. **Subject-localised bias** -- a particular LLM is consistently
   stronger / weaker than the model's prior on a *region* of items
   (e.g. medicine MCQ + small parameter count). The metadata towers
   already absorb the macro structure but they cannot capture
   fine-grained "this LLM is +0.05 above the prior on items that look
   like this neighborhood" signals.
2. **Item-neighborhood bias** -- two items that share a tiny embedding
   neighborhood (paraphrases, near-duplicates) tend to be
   miscalibrated together.

This module implements the standard Netflix-Prize-era fix: for a
query ``(s, i)`` find its top-K nearest training items in embedding
space, look up the per-(subject, neighbor) residual
``r_n = y_n - p_uncal_n`` for the *same* subject, and shift the
prediction by a similarity-weighted, shrunk average of those
residuals:

    p_cal = sigmoid(
        logit(p_uncal)
        + alpha * (
            sum_n w_n * r_n / max(eps, sum_n w_n)
        )
    )

where ``w_n`` is the kernel-weight of neighbor ``n`` (cosine sim
clipped to ``[0, 1]`` raised to a temperature) and ``alpha`` is a
single scalar shrinkage fit on a held-out split. Setting ``alpha=0``
recovers the uncalibrated model; ``alpha=1`` lets the neighborhood
fully control the residual.

Storage / runtime contract
--------------------------
The fitted calibrator is a tiny JSON blob (``alpha``, ``temperature``,
``min_weight_sum``, ``k``, ``similarity``). The *data* it consumes at
apply-time -- per-(subject, training_item) labels and uncalibrated
probabilities -- is shipped via :class:`SubjectResidualTable`, which
serializes as two sparse CSR matrices (``passrate.npz`` and
``uncal_prob.npz``) with shared row indexing. The runtime
``model.py`` re-uses the FAISS index that the existing NN-feature
training cache already ships, so this calibrator does NOT add a
separate index file.

Ensembling
----------
For a future ensemble-level neighbor calibrator, every member ships
its own ``SubjectResidualTable`` (its own per-(subj, item) p_uncal)
plus a shared item index. The ensemble calibrator can then either
(a) average per-member residual corrections before sigmoid, or
(b) fit a per-member alpha. The serialization here is forward
compatible with both.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

LOG = logging.getLogger("nn_calibration")


_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable; np.where keeps both branches finite.
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ---------------------------------------------------------------------------
# Per-(subject, training_item) residual table.
# ---------------------------------------------------------------------------


@dataclass
class SubjectResidualTable:
    """Sparse CSR storage for (subject_id, training_item_row) -> (y, p_uncal).

    ``passrate`` and ``uncal_prob`` are aligned CSR matrices of shape
    ``[n_subjects, n_training_items]`` (training items in the order the
    NN index uses; row ``i`` of the embedding matrix is item-row ``i``).
    The same sparsity pattern is shared by both -- whenever
    ``passrate.has[s, i]`` is True we also have a recorded
    ``uncal_prob[s, i]``.

    The mask is reconstructed from ``passrate.indptr/indices`` so we
    don't ship a third copy.
    """

    passrate_indptr: np.ndarray  # int64 [n_subjects + 1]
    passrate_indices: np.ndarray  # int32 [nnz]
    passrate_data: np.ndarray  # float32 [nnz]  -- y in [0, 1]
    uncal_prob_data: np.ndarray  # float32 [nnz]  -- p_uncal in [0, 1]
    n_subjects: int
    n_training_items: int

    def __post_init__(self) -> None:
        if self.passrate_data.shape != self.uncal_prob_data.shape:
            raise ValueError(
                f"passrate_data {self.passrate_data.shape} != "
                f"uncal_prob_data {self.uncal_prob_data.shape}"
            )
        if self.passrate_indices.shape != self.passrate_data.shape:
            raise ValueError(
                f"passrate_indices {self.passrate_indices.shape} != "
                f"passrate_data {self.passrate_data.shape}"
            )
        if int(self.passrate_indptr.shape[0]) != int(self.n_subjects) + 1:
            raise ValueError(
                f"passrate_indptr length {self.passrate_indptr.shape[0]} "
                f"!= n_subjects+1 ({int(self.n_subjects) + 1})"
            )

    @classmethod
    def from_rows(
        cls,
        *,
        subject_ids: Sequence[int],
        training_item_rows: Sequence[int],
        labels: Sequence[float],
        uncal_probs: Sequence[float],
        n_subjects: int,
        n_training_items: int,
    ) -> "SubjectResidualTable":
        """Build from per-row arrays.

        Multiple rows for the same ``(subject_id, training_item_row)``
        get aggregated by mean. The mean is the right reduction for
        binary labels (it equals the empirical pass-rate the existing
        NN feature table uses) and also for the uncalibrated
        probability (we want the model's average prediction over the
        rows we observed, which mean accomplishes -- the model is
        deterministic given the inputs so duplicates are rare anyway).
        """
        s = np.asarray(subject_ids, dtype=np.int64)
        r = np.asarray(training_item_rows, dtype=np.int64)
        y = np.asarray(labels, dtype=np.float32)
        p = np.asarray(uncal_probs, dtype=np.float32)
        if not (s.shape == r.shape == y.shape == p.shape):
            raise ValueError(
                f"shape mismatch: subject_ids={s.shape} "
                f"training_item_rows={r.shape} labels={y.shape} "
                f"uncal_probs={p.shape}"
            )

        if s.size == 0:
            indptr = np.zeros(int(n_subjects) + 1, dtype=np.int64)
            indices = np.zeros(0, dtype=np.int32)
            data = np.zeros(0, dtype=np.float32)
            return cls(
                passrate_indptr=indptr,
                passrate_indices=indices,
                passrate_data=data.copy(),
                uncal_prob_data=data.copy(),
                n_subjects=int(n_subjects),
                n_training_items=int(n_training_items),
            )

        # Validate ranges before scattering.
        if int(s.min()) < 0 or int(s.max()) >= int(n_subjects):
            raise ValueError(
                f"subject_ids out of range [0, {int(n_subjects)})"
            )
        if int(r.min()) < 0 or int(r.max()) >= int(n_training_items):
            raise ValueError(
                f"training_item_rows out of range [0, {int(n_training_items)})"
            )

        # Group by (s, r) using a 1-D linear key so we can use np.unique.
        flat = s.astype(np.int64) * np.int64(n_training_items) + r.astype(
            np.int64
        )
        unique, inverse, counts = np.unique(
            flat, return_inverse=True, return_counts=True
        )
        sum_y = np.zeros(unique.shape, dtype=np.float64)
        sum_p = np.zeros(unique.shape, dtype=np.float64)
        np.add.at(sum_y, inverse, y.astype(np.float64))
        np.add.at(sum_p, inverse, p.astype(np.float64))
        mean_y = (sum_y / counts).astype(np.float32)
        mean_p = (sum_p / counts).astype(np.float32)

        # Decode unique (subj, row) -> sort lexicographically for CSR.
        u_subj = (unique // np.int64(n_training_items)).astype(np.int64)
        u_row = (unique % np.int64(n_training_items)).astype(np.int32)

        # CSR: sort by subject first, then by row within each subject.
        order = np.lexsort((u_row, u_subj))
        u_subj = u_subj[order]
        u_row = u_row[order]
        mean_y = mean_y[order]
        mean_p = mean_p[order]

        # Build indptr from per-subject counts.
        indptr = np.zeros(int(n_subjects) + 1, dtype=np.int64)
        np.add.at(indptr, u_subj + 1, 1)
        np.cumsum(indptr, out=indptr)

        return cls(
            passrate_indptr=indptr,
            passrate_indices=u_row,
            passrate_data=mean_y,
            uncal_prob_data=mean_p,
            n_subjects=int(n_subjects),
            n_training_items=int(n_training_items),
        )

    # ---------------------------------------------------- query helpers

    def lookup(
        self, subject_id: int, neighbor_rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(labels, uncal_probs, mask)`` for one subject.

        ``neighbor_rows`` is a 1-D int array of training-item-row
        indices (output of FAISS / kNN). The returned arrays are
        aligned with ``neighbor_rows`` and have the same length:
        ``labels[j]`` and ``uncal_probs[j]`` are zero where
        ``mask[j] == 0``.
        """
        s = int(subject_id)
        if s < 0 or s >= int(self.n_subjects):
            n = neighbor_rows.shape[0]
            return (
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
            )
        a = int(self.passrate_indptr[s])
        b = int(self.passrate_indptr[s + 1])
        if a == b:
            n = neighbor_rows.shape[0]
            return (
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
            )
        sub_idx = self.passrate_indices[a:b]      # sorted ascending
        sub_y = self.passrate_data[a:b]
        sub_p = self.uncal_prob_data[a:b]

        # `searchsorted` over sorted ascending indices.
        nbr = np.asarray(neighbor_rows, dtype=np.int64)
        pos = np.searchsorted(sub_idx, nbr)
        # Guard against pos == len(sub_idx)
        clipped = np.clip(pos, 0, len(sub_idx) - 1)
        hit = (clipped == pos) & (sub_idx[clipped] == nbr)
        labels = np.where(hit, sub_y[clipped], 0.0).astype(np.float32)
        probs = np.where(hit, sub_p[clipped], 0.0).astype(np.float32)
        mask = hit.astype(np.float32)
        return labels, probs, mask

    # ---------------------------------------------------- serialization

    def save(self, out_dir: Path) -> Path:
        """Write to ``out_dir`` as four .npy files + one meta.json.

        Layout:
          out_dir/passrate_indptr.npy
          out_dir/passrate_indices.npy
          out_dir/passrate_data.npy
          out_dir/uncal_prob_data.npy
          out_dir/meta.json  -- {n_subjects, n_training_items, nnz}
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "passrate_indptr.npy", self.passrate_indptr)
        np.save(out_dir / "passrate_indices.npy", self.passrate_indices)
        np.save(out_dir / "passrate_data.npy", self.passrate_data)
        np.save(out_dir / "uncal_prob_data.npy", self.uncal_prob_data)
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "n_subjects": int(self.n_subjects),
                    "n_training_items": int(self.n_training_items),
                    "nnz": int(self.passrate_data.shape[0]),
                    "format_version": 1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out_dir

    @classmethod
    def load(cls, in_dir: Path) -> "SubjectResidualTable":
        in_dir = Path(in_dir)
        meta = json.loads((in_dir / "meta.json").read_text(encoding="utf-8"))
        return cls(
            passrate_indptr=np.load(in_dir / "passrate_indptr.npy").astype(
                np.int64, copy=False
            ),
            passrate_indices=np.load(in_dir / "passrate_indices.npy").astype(
                np.int32, copy=False
            ),
            passrate_data=np.load(in_dir / "passrate_data.npy").astype(
                np.float32, copy=False
            ),
            uncal_prob_data=np.load(in_dir / "uncal_prob_data.npy").astype(
                np.float32, copy=False
            ),
            n_subjects=int(meta["n_subjects"]),
            n_training_items=int(meta["n_training_items"]),
        )


# ---------------------------------------------------------------------------
# The calibrator itself.
# ---------------------------------------------------------------------------


@dataclass
class NNCalibratorState:
    """JSON-friendly state of a fitted calibrator. No numpy arrays."""

    alpha: float = 0.5
    temperature: float = 1.0
    k: int = 16
    similarity: str = "cosine"
    min_weight_sum: float = 1e-3
    similarity_floor: float = 0.0
    # When False (the default), residuals are computed in *probability
    # space*: ``r = y - p_uncal``. This is the standard Netflix-Prize
    # form for binary labels: ``r`` is bounded in ``[-1, 1]`` and the
    # signal aggregates well under similarity-weighted averaging.
    #
    # When True, residuals are computed in *logit space*. Logits of
    # binary labels saturate at ``+-log(1/eps - 1)`` after clipping,
    # which makes the per-neighbor residual extremely large and
    # sensitive to noise. The logit form is provided for callers that
    # need to chain this calibrator with a downstream temperature /
    # intercept calibrator that itself operates on logits, but it
    # should *not* be used naively on raw binary labels.
    apply_in_logit_space: bool = False
    fit_method: str = "identity"
    fit_n_val: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping | None) -> "NNCalibratorState":
        d = dict(d or {})
        return cls(
            alpha=float(d.get("alpha", 0.5)),
            temperature=float(d.get("temperature", 1.0)),
            k=int(d.get("k", 16)),
            similarity=str(d.get("similarity", "cosine")),
            min_weight_sum=float(d.get("min_weight_sum", 1e-3)),
            similarity_floor=float(d.get("similarity_floor", 0.0)),
            apply_in_logit_space=bool(d.get("apply_in_logit_space", False)),
            fit_method=str(d.get("fit_method", "identity")),
            fit_n_val=int(d.get("fit_n_val", 0)),
        )


class NNCalibrator:
    """Apply a Netflix-Prize-style residual correction.

    The class is the same in training and in the runtime template; a
    fitted instance serializes to a small JSON blob via
    :meth:`to_dict` and the data tables travel separately as a
    :class:`SubjectResidualTable` written to disk.
    """

    def __init__(self, state: NNCalibratorState | None = None) -> None:
        self.state: NNCalibratorState = state or NNCalibratorState()

    # ---------------------------------------------------- fit on val

    @classmethod
    def fit_alpha_on_val(
        cls,
        *,
        residual_table: SubjectResidualTable,
        val_subject_ids: Sequence[int],
        val_neighbor_rows: np.ndarray,        # [N_val, K] int64
        val_neighbor_sims: np.ndarray,        # [N_val, K] float32
        val_uncal_probs: Sequence[float],
        val_labels: Sequence[float],
        k: int = 16,
        similarity: str = "cosine",
        temperature: float = 1.0,
        similarity_floor: float = 0.0,
        min_weight_sum: float = 1e-3,
        apply_in_logit_space: bool = False,
        alphas: Sequence[float] | None = None,
    ) -> "NNCalibrator":
        """Sweep ``alpha`` on the validation split and pick the best NLL.

        The alpha grid defaults to a 21-point sweep over [0, 1]. We
        evaluate the binary cross-entropy (log-loss) on the validation
        rows after applying the kNN residual correction and pick the
        alpha that minimizes it. If every alpha gives a worse loss
        than the un-corrected baseline, alpha=0.0 (identity) is
        returned -- the calibrator becomes a no-op rather than
        actively making things worse.
        """
        if alphas is None:
            alphas = list(np.linspace(0.0, 1.0, 21))

        # Cache the per-row signed residual sum / weight sum so we
        # only loop once over the val set.
        s_arr = np.asarray(val_subject_ids, dtype=np.int64)
        nbrs = np.asarray(val_neighbor_rows, dtype=np.int64)
        sims = np.asarray(val_neighbor_sims, dtype=np.float32)
        if nbrs.shape != sims.shape:
            raise ValueError(
                f"shape mismatch: nbrs={nbrs.shape} sims={sims.shape}"
            )
        N = int(nbrs.shape[0])
        if N == 0:
            return cls(NNCalibratorState(
                alpha=0.0,
                temperature=float(temperature),
                k=int(k),
                similarity=str(similarity),
                min_weight_sum=float(min_weight_sum),
                similarity_floor=float(similarity_floor),
                apply_in_logit_space=bool(apply_in_logit_space),
                fit_method="empty",
                fit_n_val=0,
            ))

        if int(nbrs.shape[0]) != s_arr.shape[0]:
            raise ValueError(
                f"val_subject_ids has {s_arr.shape[0]} rows but neighbor "
                f"matrices have {nbrs.shape[0]}"
            )

        K_eff = min(int(k), int(nbrs.shape[1]))
        nbrs = nbrs[:, :K_eff]
        sims = sims[:, :K_eff]

        # Per-row weighted residual mean, in logit space if requested.
        weighted_residual = np.zeros(N, dtype=np.float64)

        weights = np.clip(sims - float(similarity_floor), 0.0, None)
        if float(temperature) != 1.0:
            weights = np.power(weights, float(temperature))
        weight_sums = np.zeros(N, dtype=np.float64)

        for i in range(N):
            ys, ps, m = residual_table.lookup(int(s_arr[i]), nbrs[i])
            if not bool(m.any()):
                continue
            w = weights[i] * m
            w_sum = float(w.sum())
            if w_sum < float(min_weight_sum):
                continue
            if apply_in_logit_space:
                # residual on logit scale: logit(y) is undefined, so
                # we use logit(neighbor_p_uncal) - logit(y_smoothed)
                # with y smoothed away from {0, 1} via the same eps.
                # For a binary label the residual is well approximated
                # by (y - p) at p ~ y, but the logit form is more
                # numerically stable when p is near 0 or 1.
                target_logit = _logit(ys)
                base_logit = _logit(ps)
                r = target_logit - base_logit
            else:
                r = ys.astype(np.float64) - ps.astype(np.float64)
            weighted_residual[i] = float((w.astype(np.float64) * r).sum() / w_sum)
            weight_sums[i] = w_sum

        y_val = np.asarray(val_labels, dtype=np.float64)
        p_val = np.asarray(val_uncal_probs, dtype=np.float64)
        p_val = np.clip(p_val, _EPS, 1.0 - _EPS)
        logit_val = np.log(p_val / (1.0 - p_val))

        baseline_nll = -float(
            (y_val * np.log(p_val) + (1.0 - y_val) * np.log(1.0 - p_val)).mean()
        )
        best_alpha = 0.0
        best_nll = baseline_nll
        for a in alphas:
            a = float(a)
            if apply_in_logit_space:
                logit_cal = logit_val + a * weighted_residual
                p_cal = 1.0 / (1.0 + np.exp(-logit_cal))
                p_cal = np.clip(p_cal, _EPS, 1.0 - _EPS)
            else:
                p_cal = np.clip(
                    p_val + a * weighted_residual, _EPS, 1.0 - _EPS
                )
            nll = -float(
                (y_val * np.log(p_cal) + (1.0 - y_val) * np.log(1.0 - p_cal)).mean()
            )
            if nll < best_nll - 1e-9:
                best_nll = nll
                best_alpha = a

        LOG.info(
            "NNCalibrator fit: alpha=%.3f val_nll=%.5f baseline_nll=%.5f "
            "(N=%d, K=%d, similarity=%s, temperature=%.2f, logit=%s)",
            best_alpha, best_nll, baseline_nll, N, K_eff, similarity,
            float(temperature), bool(apply_in_logit_space),
        )

        return cls(NNCalibratorState(
            alpha=float(best_alpha),
            temperature=float(temperature),
            k=int(K_eff),
            similarity=str(similarity),
            min_weight_sum=float(min_weight_sum),
            similarity_floor=float(similarity_floor),
            apply_in_logit_space=bool(apply_in_logit_space),
            fit_method="alpha_grid_val_nll",
            fit_n_val=int(N),
        ))

    # ---------------------------------------------------- apply

    def apply_one(
        self,
        *,
        residual_table: SubjectResidualTable,
        subject_id: int,
        neighbor_rows: np.ndarray,
        neighbor_sims: np.ndarray,
        p_uncal: float,
    ) -> float:
        """Return calibrated probability for a single query."""
        return float(
            self.apply(
                residual_table=residual_table,
                subject_ids=np.array([int(subject_id)], dtype=np.int64),
                neighbor_rows=np.asarray(neighbor_rows, dtype=np.int64).reshape(1, -1),
                neighbor_sims=np.asarray(neighbor_sims, dtype=np.float32).reshape(1, -1),
                p_uncal=np.array([float(p_uncal)], dtype=np.float32),
            )[0]
        )

    def apply(
        self,
        *,
        residual_table: SubjectResidualTable,
        subject_ids: np.ndarray,
        neighbor_rows: np.ndarray,            # [N, K]
        neighbor_sims: np.ndarray,            # [N, K]
        p_uncal: np.ndarray,                  # [N]
    ) -> np.ndarray:
        """Vectorised apply over a batch of queries."""
        st = self.state
        if float(st.alpha) == 0.0:
            return np.asarray(p_uncal, dtype=np.float32).copy()

        s_arr = np.asarray(subject_ids, dtype=np.int64)
        nbrs = np.asarray(neighbor_rows, dtype=np.int64)
        sims = np.asarray(neighbor_sims, dtype=np.float32)
        if nbrs.ndim == 1:
            nbrs = nbrs[None, :]
            sims = sims[None, :]
        K_eff = min(int(st.k), int(nbrs.shape[1]))
        nbrs = nbrs[:, :K_eff]
        sims = sims[:, :K_eff]

        weights_all = np.clip(sims - float(st.similarity_floor), 0.0, None)
        if float(st.temperature) != 1.0:
            weights_all = np.power(weights_all, float(st.temperature))

        N = int(nbrs.shape[0])
        if N == 0:
            return np.asarray(p_uncal, dtype=np.float32).copy()

        weighted_residual = np.zeros(N, dtype=np.float64)
        for i in range(N):
            ys, ps, m = residual_table.lookup(int(s_arr[i]), nbrs[i])
            if not bool(m.any()):
                continue
            w = weights_all[i] * m
            w_sum = float(w.sum())
            if w_sum < float(st.min_weight_sum):
                continue
            if st.apply_in_logit_space:
                r = (_logit(ys) - _logit(ps)).astype(np.float64)
            else:
                r = ys.astype(np.float64) - ps.astype(np.float64)
            weighted_residual[i] = float((w.astype(np.float64) * r).sum() / w_sum)

        if st.apply_in_logit_space:
            logit_uncal = _logit(np.asarray(p_uncal, dtype=np.float64))
            logit_cal = logit_uncal + float(st.alpha) * weighted_residual
            return _sigmoid(logit_cal).astype(np.float32)
        else:
            return np.clip(
                np.asarray(p_uncal, dtype=np.float64)
                + float(st.alpha) * weighted_residual,
                _EPS,
                1.0 - _EPS,
            ).astype(np.float32)

    def to_dict(self) -> dict:
        return self.state.to_dict()

    @classmethod
    def from_dict(cls, d: Mapping | None) -> "NNCalibrator":
        return cls(NNCalibratorState.from_dict(d or {}))


__all__ = [
    "NNCalibrator",
    "NNCalibratorState",
    "SubjectResidualTable",
]
