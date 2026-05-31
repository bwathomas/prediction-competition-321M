"""PCA 'tail' subspace utilities for the feature-diversity members.

Members 4 (logreg) and 5 (kNN) in the May-2026 diversification pass
operate on the *residual* (low-variance) directions of the item
embedding space rather than the coarse, variance-dominant top
components that Members 1/3 already key on.

We fit a single randomized PCA on the (unique) train item embeddings,
drop the top ``head_drop`` components (the coarse semantic axis), and
keep the next ``tail_take`` components as the 'tail' subspace. Because
the fit is fully unsupervised (no labels), reusing one global basis
across OOF folds introduces no label leakage.

The basis is small (``[D, tail_take]``) and JSON/npz-serialisable so it
can be cached with ``cache_or_compute`` and shipped in the runtime
bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _randomized_svd(
    X: np.ndarray, n_components: int, *, n_oversamples: int = 10,
    n_iter: int = 5, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (singular_values[n_components], Vt[n_components, n_features]).

    ``X`` is assumed already centered. Uses a randomized range finder
    with power iterations for numerical accuracy on slowly-decaying
    spectra.
    """
    rng = np.random.default_rng(int(seed))
    m, n = X.shape
    n_components = int(min(n_components, min(m, n)))
    l = min(n, n_components + int(n_oversamples))
    G = rng.standard_normal((n, l)).astype(np.float64)
    Y = X @ G                                   # [m, l]
    for _ in range(int(n_iter)):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y)                      # [m, l]
    B = Q.T @ X                                 # [l, n]
    _Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    return S[:n_components], Vt[:n_components]


@dataclass
class PcaTailBasis:
    """A centered PCA tail-projection: ``(x - mean) @ basis``."""

    mean: np.ndarray            # [D] float32
    basis: np.ndarray           # [D, tail_dim] float32 (columns = tail components)
    head_drop: int
    tail_take: int
    explained_variance: np.ndarray   # [tail_dim] float32 (singular-value^2 / (n-1))

    @property
    def tail_dim(self) -> int:
        return int(self.basis.shape[1])

    @property
    def d_emb(self) -> int:
        return int(self.basis.shape[0])

    def project(self, emb: np.ndarray) -> np.ndarray:
        """Project ``[m, D]`` embeddings to ``[m, tail_dim]``."""
        e = np.asarray(emb, dtype=np.float32)
        if e.ndim != 2 or int(e.shape[1]) != self.d_emb:
            raise ValueError(
                f"emb shape {e.shape} must be (m, {self.d_emb})"
            )
        return ((e - self.mean) @ self.basis).astype(np.float32, copy=False)

    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "pca_tail.npz",
            mean=self.mean.astype(np.float32),
            basis=self.basis.astype(np.float32),
            explained_variance=self.explained_variance.astype(np.float32),
        )
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "head_drop": int(self.head_drop),
                    "tail_take": int(self.tail_take),
                    "d_emb": int(self.d_emb),
                    "tail_dim": int(self.tail_dim),
                    "format_version": 1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "PcaTailBasis":
        d = Path(in_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with np.load(d / "pca_tail.npz") as npz:
            return cls(
                mean=npz["mean"].astype(np.float32, copy=False),
                basis=npz["basis"].astype(np.float32, copy=False),
                head_drop=int(meta["head_drop"]),
                tail_take=int(meta["tail_take"]),
                explained_variance=npz["explained_variance"].astype(np.float32, copy=False),
            )


def fit_pca_tail(
    emb_unique: np.ndarray,
    *,
    n_components: int = 256,
    head_drop: int = 32,
    tail_take: int = 128,
    seed: int = 0,
) -> PcaTailBasis:
    """Fit the tail-subspace basis from unique item embeddings.

    Parameters
    ----------
    emb_unique
        ``[n_items, D]`` float array of (de-duplicated) item embeddings.
    n_components
        How many top PCs to compute before slicing.
    head_drop
        Number of leading (highest-variance) PCs to discard.
    tail_take
        Number of PCs to keep after the head, forming the tail subspace.
    """
    X = np.asarray(emb_unique, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"emb_unique must be 2-D, got {X.shape}")
    n, d = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean
    want = int(min(n_components, head_drop + tail_take, min(n, d)))
    if want <= int(head_drop):
        raise ValueError(
            f"not enough components ({want}) to drop a head of {head_drop}; "
            f"reduce head_drop or provide more items/dims"
        )
    S, Vt = _randomized_svd(Xc, want, seed=seed)
    comps = Vt                                   # [want, D], rows = PCs
    hi = int(min(head_drop + tail_take, comps.shape[0]))
    tail_comps = comps[int(head_drop):hi]        # [tail_dim, D]
    tail_S = S[int(head_drop):hi]
    basis = tail_comps.T.astype(np.float32)      # [D, tail_dim]
    ev = (tail_S ** 2 / max(1, (n - 1))).astype(np.float32)
    return PcaTailBasis(
        mean=mean.astype(np.float32),
        basis=basis,
        head_drop=int(head_drop),
        tail_take=int(tail_take),
        explained_variance=ev,
    )


__all__ = ["PcaTailBasis", "fit_pca_tail"]
