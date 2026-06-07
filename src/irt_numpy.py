"""Pure-Python+numpy inference for the bagged K-dim item-IRT member (irt_bag).

The ship member is M bootstrap fits of the K-dim item-amortized IRT, averaged. Each member
(saved by the TRAIN_ALL run to ``<save>/models/full/members/m##.pt``) holds:
  theta [n_subjects, K]  (free per-subject ability)
  A     [K, D_EMB]       (item discrimination = A @ emb_std)   -- nn.Linear weight
  bw_w  [1, D_EMB], bw_b [1]   (item difficulty = bw_w @ emb_std + bw_b)
  mu, sd [1, D_EMB]      (item-embedding standardization)
Forward (matches src exp_loo_category_mlp.fit_and_predict_oof_irt_bag):
  e_std = (emb - mu) / sd
  logit = sum_k theta[sid,k] * (e_std @ A.T)[:,k]  +  (e_std @ bw_w.T + bw_b)
  p     = mean_m sigmoid(logit_m)        # average over the M bag members

Offline (needs torch, to read the .pt members):  extract_irt_bag(members_dir, out.npz)
Runtime (numpy only):  arrs = load_irt_bag("out.npz");  p = irt_bag_predict(subject_ids, item_emb, arrs)
Cold/unknown subject ids (>= n_subjects or < 0) route to row 0, matching the runtime clamp.
"""
from __future__ import annotations

import glob
import numpy as np


def extract_irt_bag(members_dir: str, out_path: str | None = None) -> dict:
    """Stack the M torch ``.pt`` members into numpy arrays (needs torch to read the pickles)."""
    import torch
    files = sorted(glob.glob(f"{members_dir}/m*.pt"))
    if not files:
        raise FileNotFoundError(f"no m*.pt members in {members_dir}")
    th, A, bww, bwb = [], [], [], []
    mu = sd = None
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=True)
        th.append(np.asarray(d["theta"], np.float32))
        A.append(np.asarray(d["A"], np.float32))
        bww.append(np.asarray(d["bw_w"], np.float32).reshape(-1))
        bwb.append(float(np.asarray(d["bw_b"]).reshape(-1)[0]))
        mu = np.asarray(d["mu"], np.float32).reshape(-1)
        sd = np.asarray(d["sd"], np.float32).reshape(-1)
    arrs = {
        "theta": np.stack(th).astype(np.float32),     # [M, n_subjects, K]
        "A": np.stack(A).astype(np.float32),          # [M, K, D]
        "bw_w": np.stack(bww).astype(np.float32),     # [M, D]
        "bw_b": np.asarray(bwb, np.float32),          # [M]
        "mu": mu, "sd": sd,                           # [D]
    }
    if out_path:
        np.savez_compressed(out_path, **arrs)
    return arrs


def load_irt_bag(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def irt_bag_predict(subject_ids, item_emb, arrs: dict) -> np.ndarray:
    """Numpy-only bagged-IRT probability. subject_ids: [N] int; item_emb: [N, D]. Returns [N] float64."""
    theta, A, bw_w, bw_b = arrs["theta"], arrs["A"], arrs["bw_w"], arrs["bw_b"]
    mu, sd = arrs["mu"], arrs["sd"]
    M, nS = theta.shape[0], theta.shape[1]
    sid = np.asarray(subject_ids, np.int64).reshape(-1)
    sid = np.where((sid >= 0) & (sid < nS), sid, 0)
    e = np.asarray(item_emb, np.float32)
    es = (e - mu) / sd                                # [N, D]
    N = e.shape[0]
    acc = np.zeros(N, np.float64)
    for m in range(M):
        aE = es @ A[m].T                              # [N, K]
        th = theta[m][sid]                            # [N, K]
        logit = (th * aE).sum(1) + es @ bw_w[m] + float(bw_b[m])
        acc += 1.0 / (1.0 + np.exp(-logit))
    return acc / float(M)


__all__ = ["extract_irt_bag", "load_irt_bag", "irt_bag_predict"]
