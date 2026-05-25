"""Member 3 of the four-member stacked ensemble: a kNN-similarity
member that runs FAISS-free at runtime.

The user spec for this member calls for retrieving the k nearest
*training* items in Qwen embedding space and predicting the
similarity-weighted average of those items' labels for the same
subject, with shrinkage toward the subject's global pass-rate and
then the global pass-rate. FAISS is unavailable at runtime, so the
neighbor structure is precomputed offline and shipped as plain
numpy arrays.

Key design decisions
--------------------
1. **PCA + quantize the shipped embeddings.** Full-D Qwen3-Embedding
   vectors are 4096 floats per item; shipping them raw would blow
   the ZIP cap. We fit PCA on the training-item embeddings (no
   leakage -- training items only), project to ``pca_dim=128``,
   normalize, and ship either fp16 or int8-quantized (per-row scale)
   embeddings. Member 1 (the IRT-MLP) sees the FULL-D embeddings via
   the encoder, so the PCA-induced quality loss is local to this
   member; the residual MLP / stacker can downweight Member 3 when
   its quantization noise hurts.

2. **Pass-rate table is dense fp16.** We ship a dense
   ``[n_subjects, n_items]`` matrix of mean labels plus a boolean
   observation mask. Subjects are typically O(100); items O(few k);
   so dense fp16 is ~MBs at most. Sparse representations would be
   bigger here because the matrix is mostly observed.

3. **Two-stage Bayesian shrinkage.** Continuous, no hard cutoffs
   (no ``min_weight_sum`` guard). The effective neighbor mass
   ``N_eff = sum(sim_i * mask_i)`` interpolates the prediction
   between the similarity-weighted neighbor mean and the subject
   prior. The subject's observation count then interpolates between
   the subject prior and the global prior. Degenerate cases (zero
   neighbors, zero subject obs) flow gracefully to the global
   passrate, never crash.

4. **PCA basis + mean shipped, query projection at runtime.** The
   query embedding arrives at full D (the encoder is unchanged); the
   runtime path projects it through the stored PCA basis before the
   top-k. This keeps the on-disk artifact compact while preserving
   Member 1's full-fidelity input.

Runtime contract
----------------
``apply_one(state, q_full_D, subject_key) -> float`` (in (eps, 1-eps))
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

LOG = logging.getLogger("knn_member")

_EPS = 1.0e-6


# ---------------------------------------------------------------------------
# State (the shipped artifact)
# ---------------------------------------------------------------------------


@dataclass
class KNNMemberState:
    """Fitted-and-shipped state of Member 3.

    Storage layout (all small + numpy):
      * ``pca_basis``         fp16 [D, pca_dim]
      * ``pca_mean``          fp16 [D]
      * ``embeddings_q``      fp16 OR int8 [N_items, pca_dim]
        (already L2-normalized at fit time so runtime cosine = dot)
      * ``embeddings_scale``  fp32 [N_items] for int8; ``None`` for fp16
      * ``passrate_dense``    fp16 [S, N_items]
      * ``passrate_mask``     bool [S, N_items]
      * ``subject_obs_count`` fp32 [S]
      * ``subject_global``    fp32 [S]
      * ``global_passrate``   fp32 scalar
      * ``item_keys``         length N_items (rarely needed at
        runtime, kept for diagnostics + audit)
      * ``subject_keys``      length S
    """

    pca_basis: np.ndarray
    pca_mean: np.ndarray
    embeddings_q: np.ndarray
    embeddings_scale: np.ndarray | None
    passrate_dense: np.ndarray
    passrate_mask: np.ndarray
    subject_obs_count: np.ndarray
    subject_global: np.ndarray
    global_passrate: float

    item_keys: tuple[str, ...]
    subject_keys: tuple[str, ...]

    k: int
    pca_dim: int
    quantization: str            # "fp16" | "int8"
    similarity: str              # "cosine"
    tau_subject: float
    tau_global: float

    # Provenance
    n_train: int
    train_loss: float
    val_loss: float

    def __post_init__(self) -> None:
        N, P = int(self.embeddings_q.shape[0]), int(self.embeddings_q.shape[1])
        S = int(self.passrate_dense.shape[0])
        if int(self.pca_basis.shape[1]) != P:
            raise ValueError(
                f"pca_basis cols {self.pca_basis.shape[1]} != pca_dim {P}"
            )
        if int(self.pca_basis.shape[0]) != int(self.pca_mean.shape[0]):
            raise ValueError(
                f"pca_basis rows {self.pca_basis.shape[0]} != "
                f"pca_mean len {self.pca_mean.shape[0]}"
            )
        if int(self.passrate_dense.shape[1]) != N:
            raise ValueError(
                f"passrate_dense cols {self.passrate_dense.shape[1]} != "
                f"embeddings_q rows {N}"
            )
        if self.passrate_mask.shape != self.passrate_dense.shape:
            raise ValueError(
                f"passrate_mask shape {self.passrate_mask.shape} != "
                f"passrate_dense shape {self.passrate_dense.shape}"
            )
        if int(self.subject_obs_count.shape[0]) != S:
            raise ValueError(
                f"subject_obs_count len {self.subject_obs_count.shape[0]} != "
                f"S {S}"
            )
        if int(self.subject_global.shape[0]) != S:
            raise ValueError(
                f"subject_global len {self.subject_global.shape[0]} != S {S}"
            )
        if int(len(self.item_keys)) != N:
            raise ValueError(
                f"item_keys len {len(self.item_keys)} != N {N}"
            )
        if int(len(self.subject_keys)) != S:
            raise ValueError(
                f"subject_keys len {len(self.subject_keys)} != S {S}"
            )
        if self.quantization == "int8":
            if self.embeddings_scale is None:
                raise ValueError("int8 quantization requires embeddings_scale")
            if int(self.embeddings_scale.shape[0]) != N:
                raise ValueError(
                    f"embeddings_scale len {self.embeddings_scale.shape[0]} != N {N}"
                )
        elif self.quantization == "fp16":
            # scale is None for fp16
            if self.embeddings_scale is not None:
                LOG.warning(
                    "embeddings_scale set with fp16 quantization; ignoring it."
                )
        else:
            raise ValueError(
                f"Unsupported quantization {self.quantization!r}; "
                "expected 'fp16' or 'int8'"
            )
        if self.similarity != "cosine":
            raise ValueError(
                f"Only similarity='cosine' supported, got {self.similarity!r}"
            )
        if not math.isfinite(float(self.global_passrate)):
            raise ValueError("global_passrate is NaN/Inf")
        if not (0.0 <= float(self.global_passrate) <= 1.0):
            raise ValueError(
                f"global_passrate {self.global_passrate} not in [0, 1]"
            )

    # ---------------- Indexing helpers ----------------

    @property
    def n_items(self) -> int:
        return int(self.embeddings_q.shape[0])

    @property
    def n_subjects(self) -> int:
        return int(self.passrate_dense.shape[0])

    def subject_index(self, subject_key: str) -> int:
        """Return the row index for ``subject_key``, or -1 if unseen.

        Linear scan over the keys tuple is fine -- subjects are
        O(100). Avoids needing to ship a dict.
        """
        # Cheap memoized lookup table built lazily on first call.
        if not hasattr(self, "_subject_lookup"):
            object.__setattr__(
                self,
                "_subject_lookup",
                {k: i for i, k in enumerate(self.subject_keys)},
            )
        return int(self._subject_lookup.get(str(subject_key), -1))

    # ---------------- I/O ----------------

    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, np.ndarray] = {
            "pca_basis": self.pca_basis.astype(np.float16),
            "pca_mean": self.pca_mean.astype(np.float16),
            "embeddings_q": self.embeddings_q,
            "passrate_dense": self.passrate_dense.astype(np.float16),
            "passrate_mask": self.passrate_mask.astype(np.bool_),
            "subject_obs_count": self.subject_obs_count.astype(np.float32),
            "subject_global": self.subject_global.astype(np.float32),
        }
        if self.quantization == "int8" and self.embeddings_scale is not None:
            save_kwargs["embeddings_scale"] = self.embeddings_scale.astype(
                np.float32
            )
        np.savez_compressed(out / "knn_state.npz", **save_kwargs)
        meta = {
            "global_passrate": float(self.global_passrate),
            "item_keys": list(self.item_keys),
            "subject_keys": list(self.subject_keys),
            "k": int(self.k),
            "pca_dim": int(self.pca_dim),
            "quantization": str(self.quantization),
            "similarity": str(self.similarity),
            "tau_subject": float(self.tau_subject),
            "tau_global": float(self.tau_global),
            "n_train": int(self.n_train),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "format_version": 1,
        }
        (out / "knn_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, in_dir: Path | str) -> "KNNMemberState":
        d = Path(in_dir)
        meta = json.loads((d / "knn_meta.json").read_text(encoding="utf-8"))
        with np.load(d / "knn_state.npz") as npz:
            pca_basis = npz["pca_basis"].astype(np.float32, copy=False)
            pca_mean = npz["pca_mean"].astype(np.float32, copy=False)
            embeddings_q = npz["embeddings_q"]
            # Don't upcast int8 here -- runtime upcasts lazily.
            passrate_dense = npz["passrate_dense"].astype(np.float32, copy=False)
            passrate_mask = npz["passrate_mask"].astype(np.bool_, copy=False)
            subject_obs_count = npz["subject_obs_count"].astype(
                np.float32, copy=False
            )
            subject_global = npz["subject_global"].astype(np.float32, copy=False)
            embeddings_scale = (
                npz["embeddings_scale"].astype(np.float32, copy=False)
                if "embeddings_scale" in npz.files
                else None
            )
        return cls(
            pca_basis=pca_basis,
            pca_mean=pca_mean,
            embeddings_q=embeddings_q,
            embeddings_scale=embeddings_scale,
            passrate_dense=passrate_dense,
            passrate_mask=passrate_mask,
            subject_obs_count=subject_obs_count,
            subject_global=subject_global,
            global_passrate=float(meta["global_passrate"]),
            item_keys=tuple(meta["item_keys"]),
            subject_keys=tuple(meta["subject_keys"]),
            k=int(meta["k"]),
            pca_dim=int(meta["pca_dim"]),
            quantization=str(meta["quantization"]),
            similarity=str(meta.get("similarity", "cosine")),
            tau_subject=float(meta["tau_subject"]),
            tau_global=float(meta["tau_global"]),
            n_train=int(meta.get("n_train", 0)),
            train_loss=float(meta.get("train_loss", 0.0)),
            val_loss=float(meta.get("val_loss", 0.0)),
        )


# ---------------------------------------------------------------------------
# Pure-numpy runtime
# ---------------------------------------------------------------------------


def _project_query(state: KNNMemberState, q_full: np.ndarray) -> np.ndarray:
    """Center, project through PCA basis, L2-normalize. Returns 1-D fp32."""
    q = np.asarray(q_full, dtype=np.float32).reshape(-1)
    if int(q.shape[0]) != int(state.pca_mean.shape[0]):
        raise ValueError(
            f"query dim {q.shape[0]} != pca_mean dim {state.pca_mean.shape[0]}"
        )
    if not np.all(np.isfinite(q)):
        # NaN/Inf in the query embedding is unrecoverable for cosine
        # similarity. Fall through to the global prior at the call
        # site by returning a zero vector (which produces zero sims
        # across all neighbors -> N_eff=0 -> shrinks to subject/global).
        q = np.zeros_like(q)
    q_centered = q - state.pca_mean.astype(np.float32, copy=False)
    q_proj = q_centered @ state.pca_basis.astype(np.float32, copy=False)
    n = float(np.linalg.norm(q_proj))
    if n < 1.0e-12:
        return q_proj
    return (q_proj / n).astype(np.float32, copy=False)


def _decode_embeddings(state: KNNMemberState) -> np.ndarray:
    """Upcast quantized embeddings to fp32. Caller is expected to
    decode at most once per ``apply_one`` call."""
    if state.quantization == "fp16":
        return state.embeddings_q.astype(np.float32, copy=False)
    if state.quantization == "int8":
        # Per-row scale: emb_fp32[i] = emb_int8[i] * scale[i]
        # The int8 representation is symmetric -- max|x| -> 127.
        if state.embeddings_scale is None:
            raise RuntimeError("int8 state missing embeddings_scale")
        return state.embeddings_q.astype(np.float32) * state.embeddings_scale[
            :, None
        ].astype(np.float32, copy=False)
    raise ValueError(f"Unsupported quantization {state.quantization!r}")


def _topk_indices_descending(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the top-k scores, sorted descending.

    Uses ``argpartition`` for O(N) selection then ``argsort`` for
    O(k log k) ordering. Stable across calls as long as ``scores``
    has no duplicates; ties broken by argsort's default behavior.
    """
    n = int(scores.shape[0])
    kk = max(1, min(int(k), n))
    if n <= kk:
        return np.argsort(-scores, kind="stable")
    part = np.argpartition(-scores, kk - 1)[:kk]
    return part[np.argsort(-scores[part], kind="stable")]


def apply_one(
    state: KNNMemberState,
    q_full: np.ndarray,
    subject_key: str,
) -> float:
    """Single-row inference -> Python ``float`` in (eps, 1-eps).

    Flow:
      1. Project + L2-normalize the query through stored PCA basis.
      2. Top-k cosine similarity over training items.
      3. Look up the subject's labels for those neighbors.
      4. Two-stage Bayesian shrinkage:
           neighbor weighted-mean -> subject prior -> global prior.
    """
    # If the subject is unknown (shouldn't happen in our cold-start
    # regime since subjects are always seen, but defensive), fall back
    # to the global prior.
    s_idx = state.subject_index(subject_key)
    if s_idx < 0 or s_idx >= state.n_subjects:
        return float(_clip_prob(state.global_passrate))

    q = _project_query(state, q_full)
    if not np.any(q):
        # Pathological query (zero norm or NaN-coerced); skip neighbor
        # lookup and shrink directly.
        return float(_clip_prob(_two_stage_shrink(
            mu_neigh=0.5,
            n_eff=0.0,
            mu_subj=float(state.subject_global[s_idx]),
            mu_glob=float(state.global_passrate),
            n_subj=float(state.subject_obs_count[s_idx]),
            tau_subject=float(state.tau_subject),
            tau_global=float(state.tau_global),
        )))

    embs = _decode_embeddings(state)
    sims = embs @ q  # [N_items]

    top_idx = _topk_indices_descending(sims, int(state.k))
    sim_top = sims[top_idx].astype(np.float32, copy=False)
    labels_top = state.passrate_dense[s_idx, top_idx].astype(
        np.float32, copy=False
    )
    masks_top = state.passrate_mask[s_idx, top_idx].astype(np.float32, copy=False)

    # Negative cosine sims contribute zero weight -- they're items
    # that point AWAY from the query, not "weakly-similar". Without
    # this clip a single strongly-anti-correlated neighbor would
    # poison the weighted mean.
    weights = np.clip(sim_top, 0.0, None) * masks_top
    n_eff = float(weights.sum())
    if n_eff > 1.0e-9:
        mu_neigh = float((weights * labels_top).sum() / n_eff)
    else:
        mu_neigh = 0.5  # placeholder; alpha_subject -> 0

    p = _two_stage_shrink(
        mu_neigh=mu_neigh,
        n_eff=n_eff,
        mu_subj=float(state.subject_global[s_idx]),
        mu_glob=float(state.global_passrate),
        n_subj=float(state.subject_obs_count[s_idx]),
        tau_subject=float(state.tau_subject),
        tau_global=float(state.tau_global),
    )
    return float(_clip_prob(p))


def apply_batch(
    state: KNNMemberState,
    queries_full: np.ndarray,
    subject_keys: Sequence[str],
) -> np.ndarray:
    """Batched inference. Useful in tests + offline OOF prediction.

    Loops :func:`apply_one` -- the inner cost is dominated by the
    matmul, which numpy already vectorizes; pre-decoding embeddings
    once and per-row cycling is fine.
    """
    if queries_full.ndim != 2:
        raise ValueError(
            f"queries_full must be 2D, got {queries_full.shape}"
        )
    B = int(queries_full.shape[0])
    if B != int(len(subject_keys)):
        raise ValueError(
            f"queries len {B} != subject_keys len {len(subject_keys)}"
        )
    out = np.empty(B, dtype=np.float32)
    for i in range(B):
        out[i] = apply_one(state, queries_full[i], subject_keys[i])
    return out


def _two_stage_shrink(
    *,
    mu_neigh: float,
    n_eff: float,
    mu_subj: float,
    mu_glob: float,
    n_subj: float,
    tau_subject: float,
    tau_global: float,
) -> float:
    """Continuous shrinkage neighbors -> subject prior -> global prior."""
    n_eff = max(float(n_eff), 0.0)
    n_subj = max(float(n_subj), 0.0)
    tau_s = max(float(tau_subject), 1.0e-9)
    tau_g = max(float(tau_global), 1.0e-9)
    alpha_subject = n_eff / (n_eff + tau_s)
    p1 = alpha_subject * mu_neigh + (1.0 - alpha_subject) * mu_subj
    alpha_global = n_subj / (n_subj + tau_g)
    p2 = alpha_global * p1 + (1.0 - alpha_global) * mu_glob
    return p2


def _clip_prob(p: float) -> float:
    if not math.isfinite(float(p)):
        return 0.5
    return float(min(max(float(p), _EPS), 1.0 - _EPS))


# ---------------------------------------------------------------------------
# Offline build
# ---------------------------------------------------------------------------


def _fit_pca(X: np.ndarray, pca_dim: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy randomized PCA. Returns (basis [D, P], mean [D]).

    For typical N (few thousand) this is fast enough that we don't need
    sklearn.decomposition.PCA. Using SVD on the centered matrix.
    """
    Xf = np.asarray(X, dtype=np.float32)
    mean = Xf.mean(axis=0).astype(np.float32)
    Xc = Xf - mean[None, :]
    # Truncated SVD via numpy: U S V^T = Xc (up to PCA_dim).
    # For wide matrices (N << D) np.linalg.svd is fine on float32.
    # full_matrices=False -> U: [N, k], S: [k], Vt: [k, D] where k=min(N, D).
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = min(int(pca_dim), int(Vt.shape[0]))
    basis = Vt[:P].T.astype(np.float32)  # [D, P]
    return basis, mean


def _quantize_int8_per_row(emb_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8 quantization. Returns (q_int8, scale_fp32)."""
    abs_max = np.max(np.abs(emb_norm), axis=1)
    # Avoid zero-row collapse (a row of all zeros stays zero).
    abs_max_safe = np.where(abs_max < 1e-12, 1.0, abs_max).astype(np.float32)
    scale = abs_max_safe / 127.0
    q = np.rint(emb_norm / scale[:, None]).astype(np.int8)
    return q, scale.astype(np.float32)


def fit_knn_member(
    *,
    item_keys: Sequence[str],
    item_embeddings: np.ndarray,           # [N, D] fp32 full-fidelity
    subject_keys: Sequence[str],
    passrate_dense: np.ndarray | None = None,    # [S, N] fp32 if precomputed
    passrate_mask: np.ndarray | None = None,     # [S, N] bool/0-1 if precomputed
    train_pairs: Sequence[tuple[str, str, float]] | None = None,
    pca_dim: int = 128,
    quantization: str = "fp16",
    k: int = 16,
    tau_subject: float = 2.0,
    tau_global: float = 50.0,
    seed: int = 0,
) -> KNNMemberState:
    """Build Member 3.

    The pass-rate table can be passed two ways:
      * ``passrate_dense`` + ``passrate_mask`` (already aggregated to
        [S, N]); or
      * ``train_pairs`` (sequence of ``(subject_key, item_key, label)``)
        which we then aggregate.
    """
    item_keys_list = [str(k) for k in item_keys]
    subject_keys_list = [str(s) for s in subject_keys]
    N = len(item_keys_list)
    S = len(subject_keys_list)

    if int(item_embeddings.shape[0]) != N:
        raise ValueError(
            f"item_embeddings rows {item_embeddings.shape[0]} != "
            f"item_keys len {N}"
        )
    if quantization not in ("fp16", "int8"):
        raise ValueError(f"Unsupported quantization {quantization!r}")

    LOG.info(
        "knn_member: fitting on N=%d items, D=%d, S=%d subjects, "
        "pca_dim=%d, quantization=%s, k=%d",
        N,
        int(item_embeddings.shape[1]),
        S,
        pca_dim,
        quantization,
        k,
    )

    # ---- PCA ----
    basis, mean = _fit_pca(
        item_embeddings.astype(np.float32, copy=False),
        pca_dim=int(pca_dim),
        seed=int(seed),
    )
    Xc = item_embeddings.astype(np.float32, copy=False) - mean[None, :]
    Xp = (Xc @ basis).astype(np.float32, copy=False)
    norms = np.linalg.norm(Xp, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-12, 1.0, norms).astype(np.float32)
    Xn = (Xp / norms_safe).astype(np.float32, copy=False)

    if quantization == "fp16":
        embeddings_q = Xn.astype(np.float16)
        embeddings_scale = None
    else:
        embeddings_q, embeddings_scale = _quantize_int8_per_row(Xn)

    # ---- Pass-rate table ----
    item_to_idx = {k: i for i, k in enumerate(item_keys_list)}
    subj_to_idx = {s: i for i, s in enumerate(subject_keys_list)}
    if passrate_dense is None or passrate_mask is None:
        if train_pairs is None:
            raise ValueError(
                "Must provide either (passrate_dense, passrate_mask) or "
                "train_pairs."
            )
        pr_sum = np.zeros((S, N), dtype=np.float64)
        pr_count = np.zeros((S, N), dtype=np.float64)
        n_pairs = 0
        for s_key, i_key, label in train_pairs:
            si = subj_to_idx.get(str(s_key))
            ii = item_to_idx.get(str(i_key))
            if si is None or ii is None:
                continue
            pr_sum[si, ii] += float(label)
            pr_count[si, ii] += 1.0
            n_pairs += 1
        if n_pairs == 0:
            raise ValueError("train_pairs produced zero usable rows")
        passrate_dense_arr = np.where(
            pr_count > 0, pr_sum / np.maximum(pr_count, 1.0), 0.0
        ).astype(np.float32)
        passrate_mask_arr = (pr_count > 0).astype(np.bool_)
    else:
        passrate_dense_arr = np.asarray(passrate_dense, dtype=np.float32)
        passrate_mask_arr = np.asarray(passrate_mask, dtype=np.bool_)
        if passrate_dense_arr.shape != (S, N):
            raise ValueError(
                f"passrate_dense shape {passrate_dense_arr.shape} != "
                f"({S}, {N})"
            )
        if passrate_mask_arr.shape != (S, N):
            raise ValueError(
                f"passrate_mask shape {passrate_mask_arr.shape} != "
                f"({S}, {N})"
            )

    # Per-subject stats: obs count + mean over labeled cells.
    subject_obs_count = passrate_mask_arr.astype(np.float32).sum(axis=1)
    obs_total = (passrate_dense_arr * passrate_mask_arr.astype(np.float32)).sum(axis=1)
    subject_global = np.where(
        subject_obs_count > 0,
        obs_total / np.maximum(subject_obs_count, 1.0),
        0.5,
    ).astype(np.float32)
    total_obs = float(passrate_mask_arr.sum())
    total_pos = float(
        (passrate_dense_arr * passrate_mask_arr.astype(np.float32)).sum()
    )
    if total_obs > 0:
        global_passrate = float(total_pos / total_obs)
    else:
        global_passrate = 0.5

    state = KNNMemberState(
        pca_basis=basis,
        pca_mean=mean,
        embeddings_q=embeddings_q,
        embeddings_scale=embeddings_scale,
        passrate_dense=passrate_dense_arr,
        passrate_mask=passrate_mask_arr,
        subject_obs_count=subject_obs_count,
        subject_global=subject_global,
        global_passrate=global_passrate,
        item_keys=tuple(item_keys_list),
        subject_keys=tuple(subject_keys_list),
        k=int(k),
        pca_dim=int(pca_dim),
        quantization=str(quantization),
        similarity="cosine",
        tau_subject=float(tau_subject),
        tau_global=float(tau_global),
        n_train=int(N),
        train_loss=0.0,
        val_loss=0.0,
    )
    LOG.info(
        "knn_member: built. global_passrate=%.4f, mean subject_obs=%.1f, "
        "embeddings dtype=%s shape=%s",
        state.global_passrate,
        float(subject_obs_count.mean()),
        state.embeddings_q.dtype,
        tuple(state.embeddings_q.shape),
    )
    return state


# ---------------------------------------------------------------------------
# Reference (FAISS-equivalent) top-k for offline parity tests
# ---------------------------------------------------------------------------


def reference_topk_full(
    item_embeddings_full: np.ndarray,    # [N, D] fp32 (full-fidelity, normed)
    queries_full: np.ndarray,            # [B, D]
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-fidelity cosine top-k. Used in tests to check that our
    PCA+quantize-then-search recovers most of the same neighbors."""
    embs = np.asarray(item_embeddings_full, dtype=np.float32)
    q = np.asarray(queries_full, dtype=np.float32)
    # Normalize both sides for cosine.
    n_e = np.linalg.norm(embs, axis=1, keepdims=True)
    n_e = np.where(n_e < 1e-12, 1.0, n_e)
    embs_n = embs / n_e
    n_q = np.linalg.norm(q, axis=1, keepdims=True)
    n_q = np.where(n_q < 1e-12, 1.0, n_q)
    q_n = q / n_q
    sims = q_n @ embs_n.T
    out_idx = np.empty((q.shape[0], k), dtype=np.int64)
    out_sim = np.empty((q.shape[0], k), dtype=np.float32)
    for i in range(q.shape[0]):
        out_idx[i] = _topk_indices_descending(sims[i], k)
        out_sim[i] = sims[i, out_idx[i]]
    return out_idx, out_sim


__all__ = [
    "KNNMemberState",
    "apply_one",
    "apply_batch",
    "fit_knn_member",
    "reference_topk_full",
]
