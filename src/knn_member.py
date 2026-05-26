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
    """Upcast quantized embeddings to fp32, caching the result on the
    state so subsequent calls are zero-cost.

    Without caching, every ``apply_one`` call rebuilds the full
    ``[N_items, pca_dim]`` fp32 array from the int8 representation
    (or upcasts the fp16). For Codabench batch eval that adds up
    fast: ~50ms per call on 100k items at pca_dim=128. The cache
    is lazily populated and stored as an attribute on the dataclass
    via ``object.__setattr__`` (so it survives even if the dataclass
    is later marked frozen).

    Memory cost: ``N_items * pca_dim * 4`` bytes (~50 MB at the
    reference dimensions). The fp16 path returns a fp16 view that
    we then upcast on first decode -- but only once.
    """
    cached = getattr(state, "_decoded_emb_cache", None)
    if cached is not None:
        return cached
    if state.quantization == "fp16":
        decoded = np.ascontiguousarray(
            state.embeddings_q.astype(np.float32, copy=True)
        )
    elif state.quantization == "int8":
        if state.embeddings_scale is None:
            raise RuntimeError("int8 state missing embeddings_scale")
        decoded = np.ascontiguousarray(
            state.embeddings_q.astype(np.float32)
            * state.embeddings_scale[:, None].astype(np.float32, copy=False)
        )
    else:
        raise ValueError(f"Unsupported quantization {state.quantization!r}")
    object.__setattr__(state, "_decoded_emb_cache", decoded)
    return decoded


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


def _project_queries_batch(
    state: KNNMemberState, queries_full: np.ndarray
) -> np.ndarray:
    """Batched version of :func:`_project_query` -- centers, projects
    through PCA basis, and L2-normalizes ``[B, D]`` queries to
    ``[B, pca_dim]`` fp32 in one shot.

    Non-finite rows are coerced to zero before normalization, matching
    the per-row guard in :func:`_project_query` (zero-norm queries
    later flow through to the subject prior at the shrinkage stage).
    """
    Q = np.asarray(queries_full, dtype=np.float32)
    if Q.ndim != 2:
        raise ValueError(f"queries_full must be 2D, got {Q.shape}")
    if int(Q.shape[1]) != int(state.pca_mean.shape[0]):
        raise ValueError(
            f"query dim {Q.shape[1]} != pca_mean dim {state.pca_mean.shape[0]}"
        )
    finite_mask = np.all(np.isfinite(Q), axis=1)
    if not finite_mask.all():
        Q = np.where(finite_mask[:, None], Q, 0.0).astype(np.float32, copy=False)
    Qc = Q - state.pca_mean.astype(np.float32, copy=False)[None, :]
    Qp = Qc @ state.pca_basis.astype(np.float32, copy=False)
    norms = np.linalg.norm(Qp, axis=1)
    safe = np.where(norms < 1.0e-12, 1.0, norms).astype(np.float32)
    Qp = Qp / safe[:, None]
    # Restore zero rows where the original query was zero / non-finite.
    Qp = np.where(norms[:, None] < 1.0e-12, 0.0, Qp).astype(np.float32, copy=False)
    return Qp


def _topk_indices_descending_batch(
    scores: np.ndarray, k: int
) -> np.ndarray:
    """Vectorized top-k along ``axis=1``. Returns ``[B, k]`` int64
    indices sorted descending by score.

    Uses ``argpartition`` for O(N) selection then ``argsort`` over
    the small partition slice for the final ordering. Same numerical
    contract as :func:`_topk_indices_descending` (per-row).
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.shape}")
    B, N = int(scores.shape[0]), int(scores.shape[1])
    kk = max(1, min(int(k), N))
    if N <= kk:
        # Full sort: return all columns in descending order.
        order = np.argsort(-scores, axis=1, kind="stable")
        return order.astype(np.int64, copy=False)
    # axis=1 argpartition picks the top-k indices (unsorted).
    part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]  # [B, kk]
    # Gather the partition's scores, sort within each row.
    part_scores = np.take_along_axis(scores, part, axis=1)  # [B, kk]
    sort_idx = np.argsort(-part_scores, axis=1, kind="stable")  # [B, kk]
    out = np.take_along_axis(part, sort_idx, axis=1)  # [B, kk]
    return out.astype(np.int64, copy=False)


def _try_torch_cuda():  # noqa: ANN201 (private helper, return type intentional)
    """Return a tuple ``(torch, device)`` if torch+CUDA are available, else ``None``."""
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            return torch, torch.device("cuda")
    except Exception:
        return None
    return None


def apply_batch(
    state: KNNMemberState,
    queries_full: np.ndarray,
    subject_keys: Sequence[str],
    *,
    chunk_size: int | None = None,
    use_gpu: bool | None = None,
    progress: bool = False,
) -> np.ndarray:
    """Vectorized batched inference. Returns ``[B]`` float32 probabilities.

    Numerically identical to looping :func:`apply_one` row-by-row
    (within fp32 jitter on the matmul reordering). The vectorization
    pulls every Python-loop hot spot out:

      * Query projection: one ``[B, D] @ [D, P]`` matmul instead of
        ``B`` separate ``[1, D] @ [D, P]`` calls.
      * Decoded-embedding lookup: cached on the state (see
        :func:`_decode_embeddings`); the per-chunk matmul is
        ``[chunk, P] @ [P, N_items]``.
      * Top-k: vectorized ``argpartition`` + ``argsort`` along axis 1.
      * Two-stage shrinkage: pure numpy on ``[B]`` arrays.

    Memory + speed knobs:

      * ``chunk_size``: rows of ``queries_full`` processed at once.
        The peak working set is ``chunk_size * N_items * 4`` bytes
        for the similarity matrix. ``None`` (default) auto-picks a
        chunk that keeps that under ~4 GB; at N_items=300k that's
        ~3500 rows. Pass an explicit value to tune for your RAM.
        At B=266k, N_items=296k, the unchunked sims matrix would be
        315 GB -- so chunking is mandatory at production scale.

      * ``use_gpu``: if ``None`` (default), auto-detects torch+CUDA
        for B >= 4096 (the matmul speedup pays off at that scale).
        If ``True``, requires torch+CUDA and raises if unavailable.
        If ``False``, forces the pure-numpy CPU path (matches
        :func:`apply_one` bit-for-bit subject to fp32 jitter). The
        runtime path :func:`apply_one` never uses torch/GPU, so the
        shipped artifact has no GPU dependency; this knob only
        affects the offline OOF / val-scoring hot path.

      * ``progress``: if True, wrap the per-chunk loop in a tqdm bar.
        Useful when scoring 200k+ rows so you can watch progress.

    The runtime entry point :func:`apply_one` is unaffected (it is
    pure numpy and never imports torch).
    """
    Q = np.asarray(queries_full, dtype=np.float32)
    if Q.ndim != 2:
        raise ValueError(f"queries_full must be 2D, got {Q.shape}")
    B = int(Q.shape[0])
    if B != int(len(subject_keys)):
        raise ValueError(
            f"queries len {B} != subject_keys len {len(subject_keys)}"
        )
    if B == 0:
        return np.empty(0, dtype=np.float32)

    # Subject indices (-1 for unseen). We'll handle unseen rows after
    # the matmul by overriding their predictions with the global prior.
    s_ids = np.fromiter(
        (state.subject_index(str(k)) for k in subject_keys),
        dtype=np.int64,
        count=B,
    )

    # Project + L2-normalize all queries in one matmul (cheap; output
    # is [B, pca_dim] which is small).
    Qp = _project_queries_batch(state, Q)  # [B, P]
    zero_query_all = np.all(Qp == 0.0, axis=1)  # zero-norm queries

    embs = _decode_embeddings(state)            # [N_items, P]
    N_items = int(embs.shape[0])
    P = int(embs.shape[1])
    K = int(state.k)

    # ---------------- Chunk size auto-pick ----------------
    if chunk_size is None:
        # Bound the per-chunk sims matrix to ~4 GB float32.
        max_bytes = 4 * (1024 ** 3)
        per_row_bytes = max(N_items * 4, 1)
        chunk_size = max(1, min(B, int(max_bytes // per_row_bytes)))
    else:
        chunk_size = max(1, min(B, int(chunk_size)))

    # ---------------- Backend selection ----------------
    backend = "numpy"
    torch_ctx = None
    if use_gpu is True:
        torch_ctx = _try_torch_cuda()
        if torch_ctx is None:
            raise RuntimeError(
                "use_gpu=True but torch/CUDA is unavailable. Pass use_gpu=False "
                "to force the CPU path or use_gpu=None to auto-detect."
            )
    elif use_gpu is None and B >= 4096:
        # Auto-detect: only try GPU when the batch is large enough that
        # the H<->D copy overhead pays off.
        torch_ctx = _try_torch_cuda()
    if torch_ctx is not None:
        backend = "torch_cuda"

    # Pre-cast embeddings once. Keeping the transposed view here means
    # the per-chunk matmul is a contiguous [chunk, P] @ [P, N_items].
    embs_T_np = np.ascontiguousarray(embs.T, dtype=np.float32)  # [P, N_items]
    if backend == "torch_cuda":
        torch_mod, device = torch_ctx  # type: ignore[misc]
        embs_T_dev = torch_mod.from_numpy(embs_T_np).to(device, non_blocking=True)
    else:
        embs_T_dev = None

    LOG.debug(
        "apply_batch: B=%d N_items=%d P=%d K=%d backend=%s chunk_size=%d",
        B, N_items, P, K, backend, chunk_size,
    )

    # ---------------- Per-chunk top-k similarity ----------------
    top_idx = np.empty((B, K), dtype=np.int64)
    sim_top = np.empty((B, K), dtype=np.float32)
    chunk_iter = range(0, B, chunk_size)
    if progress:
        try:
            from tqdm.auto import tqdm
            chunk_iter = tqdm(
                list(chunk_iter), desc="knn batch", unit="chunk",
                leave=False,
            )
        except Exception:
            pass

    for start in chunk_iter:
        end = min(B, start + chunk_size)
        Qp_chunk = Qp[start:end]                          # [c, P]
        if backend == "torch_cuda":
            Qp_t = torch_mod.from_numpy(Qp_chunk).to(device)  # type: ignore[arg-type]
            sims_t = Qp_t @ embs_T_dev                    # [c, N_items]
            # Top-k on GPU is much faster than copying back first.
            sim_t, idx_t = torch_mod.topk(sims_t, k=K, dim=1, largest=True, sorted=True)
            top_idx[start:end] = idx_t.detach().cpu().numpy().astype(np.int64, copy=False)
            sim_top[start:end] = sim_t.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            sims_chunk = Qp_chunk @ embs_T_np             # [c, N_items]
            ti_chunk = _topk_indices_descending_batch(sims_chunk, K)
            top_idx[start:end] = ti_chunk
            sim_top[start:end] = np.take_along_axis(
                sims_chunk, ti_chunk, axis=1
            ).astype(np.float32, copy=False)

    if backend == "torch_cuda":
        # Free GPU memory promptly so the caller can immediately use the
        # GPU for something else (e.g. encoder forward passes).
        del embs_T_dev
        try:
            torch_mod.cuda.empty_cache()  # type: ignore[union-attr]
        except Exception:
            pass

    # ---------------- Gather labels + masks (CPU, vectorized) ----------------
    s_safe = np.where(s_ids >= 0, s_ids, 0).astype(np.int64, copy=False)
    labels_top = state.passrate_dense[s_safe[:, None], top_idx].astype(
        np.float32, copy=False
    )
    masks_top = state.passrate_mask[s_safe[:, None], top_idx].astype(
        np.float32, copy=False
    )

    # Cosine sims clipped at 0 then weighted by mask.
    weights = np.clip(sim_top, 0.0, None) * masks_top  # [B, K]
    n_eff = weights.sum(axis=1)                        # [B]
    has_support = n_eff > 1.0e-9
    mu_neigh = np.where(
        has_support,
        np.divide(
            (weights * labels_top).sum(axis=1),
            np.where(has_support, n_eff, 1.0),
        ),
        0.5,
    ).astype(np.float64, copy=False)

    # Vectorized two-stage shrinkage. Match the scalar formula:
    #   alpha_subject = n_eff / (n_eff + tau_subject)
    #   p1            = a * mu_neigh + (1 - a) * mu_subj
    #   alpha_global  = n_subj / (n_subj + tau_global)
    #   p2            = b * p1 + (1 - b) * mu_glob
    tau_s = max(float(state.tau_subject), 1.0e-9)
    tau_g = max(float(state.tau_global), 1.0e-9)
    n_subj = state.subject_obs_count[s_safe].astype(np.float64, copy=False)
    mu_subj = state.subject_global[s_safe].astype(np.float64, copy=False)
    mu_glob = float(state.global_passrate)

    a = n_eff.astype(np.float64) / (n_eff.astype(np.float64) + tau_s)
    p1 = a * mu_neigh + (1.0 - a) * mu_subj
    b = n_subj / (n_subj + tau_g)
    p_out = b * p1 + (1.0 - b) * mu_glob

    # Apply the same fallbacks the per-row path uses.
    # Zero-norm query: skip the neighbor stage, shrink via subject -> global only.
    if zero_query_all.any():
        a_zero = np.zeros_like(a)
        p1_zero = a_zero * 0.5 + (1.0 - a_zero) * mu_subj
        p_zero = b * p1_zero + (1.0 - b) * mu_glob
        p_out = np.where(zero_query_all, p_zero, p_out)

    # Unknown subject -> global prior.
    p_out = np.where(s_ids >= 0, p_out, mu_glob)

    # Defensive clamp to (eps, 1-eps) and replace NaNs with 0.5.
    p_out = np.where(np.isfinite(p_out), p_out, 0.5)
    p_out = np.clip(p_out, _EPS, 1.0 - _EPS)
    return p_out.astype(np.float32, copy=False)


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


def _fit_pca(
    X: np.ndarray,
    pca_dim: int,
    *,
    seed: int = 0,
    n_oversamples: int = 10,
    n_iter: int = 2,
    max_pca_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncated PCA via randomized SVD. Returns (basis [D, P], mean [D]).

    Halko-Martinsson-Tropp randomized range finder + power iteration +
    small SVD on the projected matrix. For wide matrices with
    ``N >> pca_dim`` (e.g. 300k items at D=4096, pca_dim=128) this is
    20-100x faster than full ``np.linalg.svd`` because we only solve
    for the top ``pca_dim`` directions instead of all ``min(N, D)``.

    Cost (per matmul): ``N * D * (pca_dim + n_oversamples)`` fmas.
    With 2 power iterations (default) we do ~5 matmuls of that size,
    vs full SVD's ``O(N * D^2)``. At the reference shape (N=300k,
    D=4096, pca_dim=128) that's ~7 min -> ~30 sec.

    Knobs:
      * ``n_oversamples`` (default 10): extra random columns to
        improve top-component accuracy. 10 is the Halko et al.
        recommendation; rarely worth tuning.
      * ``n_iter`` (default 2): number of power iterations. 2 is
        enough for the spectrum we see in Qwen embeddings; bump to 4
        if you observe the resulting basis is unstable across runs.
      * ``max_pca_samples`` (default None): if set, fit PCA on a
        random subsample of ``max_pca_samples`` rows. The top-128
        directions stabilize fast with sample size, so e.g. 50k of
        300k items gives essentially the same basis 10x faster. The
        mean is still computed on the full ``X`` so test-time
        centering is unaffected.

    Numerical contract: the returned basis is approximately equal to
    full-SVD's top-``pca_dim`` right singular vectors, up to a sign
    flip per column and ~1e-3 relative jitter. Downstream cosine
    similarity is invariant to sign, so this doesn't affect the
    quantized neighbor search.
    """
    Xf = np.asarray(X, dtype=np.float32)
    mean = Xf.mean(axis=0).astype(np.float32)

    N = int(Xf.shape[0])
    if max_pca_samples is not None and N > int(max_pca_samples):
        # Subsample for the SVD only. Mean is from the full X so
        # runtime centering of any future query stays consistent.
        rng_sub = np.random.default_rng(int(seed) ^ 0xCAFE)
        sub_idx = rng_sub.choice(N, size=int(max_pca_samples), replace=False)
        Xc = (Xf[sub_idx] - mean[None, :]).astype(np.float32, copy=False)
        LOG.info(
            "knn_member: PCA subsample %d / %d items for SVD fit",
            int(max_pca_samples), N,
        )
    else:
        Xc = (Xf - mean[None, :]).astype(np.float32, copy=False)

    rng = np.random.default_rng(int(seed))
    Ns, D = int(Xc.shape[0]), int(Xc.shape[1])
    p_target = int(pca_dim)
    p = min(p_target + int(n_oversamples), D, Ns)

    # Randomized range finder.
    Omega = rng.standard_normal((D, p)).astype(np.float32)
    Y = Xc @ Omega                                 # [Ns, p]

    # Power iteration -- raises the small singular values relative to
    # the top-pca_dim ones, sharpening the recovered basis.
    for _ in range(int(n_iter)):
        Q, _ = np.linalg.qr(Y)                     # [Ns, p]
        Z = Xc.T @ Q                               # [D, p]
        Qz, _ = np.linalg.qr(Z)                    # [D, p]
        Y = Xc @ Qz                                # [Ns, p]

    Q, _ = np.linalg.qr(Y)                         # [Ns, p]

    # Small SVD on the [p, D] projected matrix.
    B = Q.T @ Xc                                   # [p, D]
    _, _, Vt = np.linalg.svd(B, full_matrices=False)  # Vt: [p, D]

    P = min(int(p_target), int(Vt.shape[0]))
    basis = Vt[:P].T.astype(np.float32)            # [D, P]
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
