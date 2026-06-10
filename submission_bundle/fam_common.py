"""Shared numpy runtime for the 3-family ensemble submission (no external packages).

Everything here is a faithful vendored copy of the training-time feature/member code:
  * cluster blocks   <- aide/features/cluster_fast.py (+ fixed soft-responsibility scale:
                        training used the query CHUNK's median sqdist as temperature, which
                        is batch-dependent; we ship the train-set global median instead)
  * nn blocks        <- aide/features/derive_nn.py aggregation over a PQ-ADC kNN
                        (src/pq_index.py); deep retrieval buffer because PQ sims saturate
                        the alias threshold across near-duplicate groups
  * passrate         <- aide/features/passrate.py CsrPassrate (numpy-only read path)
  * mlp forward      <- src/mlp_member.py apply_batch (gated GLU MLP, z-scored dense)
  * forest           <- src/tree_numpy.py forest_predict
  * linear stacks    <- non-neg logit blends (weights fit offline, shipped in meta)

Dense feature layout (column order is the TRAINING order; counts must match):
  centroid_distance(256) | cluster_geometry(9) | nn_geometry(3) | item_cluster(257) |
  nn_label_derivatives(13) | cluster_passrate(1) | cluster_subject(3) | counts_subject(1)
  = 543; trees additionally get item_emb_pca(PDIM) appended.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

EPS = 1e-7
KS = (4, 8, 32, 64)
MAXK = 64
NN_ALIAS_EPS = 1e-6
NN_SEARCH_BUFFER = 512   # deep: PQ sims saturate >= 1-alias_eps on near-duplicate groups

GROUP_ORDER = ["centroid_distance", "cluster_geometry", "nn_geometry", "item_cluster",
               "nn_label_derivatives", "cluster_passrate", "cluster_subject",
               "counts_subject"]


# ---------------------------------------------------------------------------------
# keys / text (must match training prep + trc5 conventions exactly)
# ---------------------------------------------------------------------------------
def normalize_condition(value) -> str:
    s = str(value) if value is not None else ""
    return "none" if s.strip().lower() in ("", "nan", "none", "null") else s


def stable_sha256(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def item_key_for(benchmark: str, condition: str, item_content: str) -> str:
    return stable_sha256(benchmark, normalize_condition(condition), item_content)


def subject_key_for(subject_content: str) -> str:
    return stable_sha256(subject_content)


def item_text_for(benchmark: str, condition: str, item_content: str,
                  passage_prefix: str = "") -> str:
    body = f"Benchmark: {benchmark}\nCondition: {condition}\nItem: {item_content}"
    return f"{passage_prefix}{body}" if passage_prefix else body


def unit_rows(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    z = np.clip(np.asarray(z, np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def linear_blend(probs, weights, bias) -> float:
    z = float(bias)
    for p, w in zip(probs, weights):
        z += float(w) * float(_logit(p))
    return float(np.clip(_sigmoid(z), EPS, 1 - EPS))


# ---------------------------------------------------------------------------------
# CSR passrate (vendored aide/features/passrate.py; numpy-only)
# ---------------------------------------------------------------------------------
class CsrPassrate:
    def __init__(self, n_subjects, n_items, indptr, indices, data):
        self.n_subjects = int(n_subjects)
        self.n_items = int(n_items)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.float64)
        self._tot_sum = float(self.data.sum())
        self._tot_cnt = float(self.data.size)
        subj_nz = np.repeat(np.arange(self.n_subjects), np.diff(self.indptr))
        self._pair_keys = subj_nz.astype(np.int64) * self.n_items + self.indices
        order = np.argsort(self._pair_keys, kind="stable")
        self._pair_keys = self._pair_keys[order]
        self._pair_vals = self.data[order]

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(int(z["n_subjects"]), int(z["n_items"]),
                       z["indptr"], z["indices"], z["data"])

    def global_mean(self) -> float:
        return self._tot_sum / self._tot_cnt if self._tot_cnt > 0 else 0.0

    def gather_pairs(self, subject_rows, item_cols) -> np.ndarray:
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
        item_to_cluster = np.asarray(item_to_cluster, dtype=np.int64)
        cl_nz = item_to_cluster[self.indices]
        subj_nz = np.repeat(np.arange(self.n_subjects), np.diff(self.indptr))
        csum = np.zeros(n_clusters)
        ccnt = np.zeros(n_clusters)
        np.add.at(csum, cl_nz, self.data)
        np.add.at(ccnt, cl_nz, 1.0)
        gm = self.global_mean()
        difficulty = np.where(ccnt > 0, csum / np.maximum(ccnt, 1.0), gm)
        ssum = np.zeros((self.n_subjects, n_clusters))
        scnt = np.zeros((self.n_subjects, n_clusters))
        np.add.at(ssum, (subj_nz, cl_nz), self.data)
        np.add.at(scnt, (subj_nz, cl_nz), 1.0)
        smean = np.where(scnt > 0, ssum / np.maximum(scnt, 1.0), np.nan)
        return difficulty, smean, scnt


# ---------------------------------------------------------------------------------
# cluster geometry/label features (vendored aide/features/cluster_fast.py)
# ---------------------------------------------------------------------------------
def _sqdist(X, C):
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    return (np.sum(X * X, axis=1)[:, None] - 2 * X @ C.T
            + np.sum(C * C, axis=1)[None, :])


def _soft_responsibility(sqd, scale):
    """Softmax over -sqd/scale. Training used the query CHUNK's median sqd as scale;
    runtime ships the train-set global median (fixed) for batch-size invariance."""
    z = -np.asarray(sqd, np.float64) / (float(scale) + 1e-9)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cluster_geometry_block(query_emb_unit, fine, coarse, sizes, sqd_scale):
    """centroid_distance(256) + cluster_geometry(9) + item_cluster(257) for query rows.
    Returns (cd [n,256], geo [n,9], oh [n,257], q_fine [n])."""
    Kf = fine.shape[0]
    q_sqd = _sqdist(query_emb_unit, fine)
    q_fine = q_sqd.argmin(1)
    resp = _soft_responsibility(q_sqd, sqd_scale)
    nq = q_sqd.shape[0]
    sorted_resp = -np.sort(-resp, axis=1)
    top = np.zeros((nq, 3))
    top[:, :min(3, Kf)] = sorted_resp[:, :min(3, Kf)]
    margin = top[:, 0] - top[:, 1]
    entropy = -np.where(resp > 0, resp * np.log(np.where(resp > 0, resp, 1.0)), 0.0).sum(1)
    typicality = -q_sqd[np.arange(nq), q_fine]
    size_log1p = np.log1p(np.asarray(sizes, np.float64)[q_fine])
    coarse_id = _sqdist(query_emb_unit, coarse).argmin(1)
    geo = np.column_stack([top[:, 0], top[:, 1], top[:, 2], margin, entropy, typicality,
                           size_log1p, coarse_id.astype(float), q_fine.astype(float)])
    oh = np.zeros((nq, Kf), dtype=np.float32)
    oh[np.arange(nq), q_fine] = 1.0
    oh_full = np.column_stack([oh, q_fine.astype(float)])
    return (q_sqd.astype(np.float32), geo.astype(np.float32),
            oh_full.astype(np.float32), q_fine, resp)


def cluster_label_cols(q_fine, resp, s_row, difficulty, smean, scnt):
    """cluster_passrate(1) + cluster_subject(3) for ONE (subject,item)."""
    k = int(q_fine)
    diff_r = float(difficulty[k])
    valid = s_row >= 0
    if valid:
        cnt_r = float(scnt[s_row, k])
        sm = smean[s_row, k]
        subj_pass = float(sm) if np.isfinite(sm) else diff_r
    else:
        cnt_r = 0.0
        subj_pass = diff_r
    gap = subj_pass - diff_r
    obs_log1p = np.log1p(cnt_r)
    if valid:
        mask = scnt[s_row] > 0
        sm_row = np.where(mask, np.nan_to_num(smean[s_row], nan=0.0), 0.0)
        num = float((resp * sm_row).sum())
        den = float((resp * mask).sum())
        soft = num / max(den, 1e-12) if den > 0 else 0.0
    else:
        soft = 0.0
    return np.array([diff_r], np.float32), np.array([gap, obs_log1p, soft], np.float32)


# ---------------------------------------------------------------------------------
# PQ index + ADC kNN (vendored src/pq_index.py)
# ---------------------------------------------------------------------------------
class PqIndex:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.codebook = z["codebook"].astype(np.float32)   # [M,256,ds]
            self.codes = z["codes"]                            # [N,M] uint8
            self.item_keys = z["item_keys"].astype(str)
            self.M, self.ds, self.D = int(z["M"]), int(z["ds"]), int(z["D"])
        self.N = self.codes.shape[0]
        self._codes_i32 = None
        self._torch = None

    def _try_torch(self):
        if self._torch is None:
            try:
                import torch
                if torch.cuda.is_available():
                    self._torch = {
                        "torch": torch,
                        "codes": torch.from_numpy(
                            np.ascontiguousarray(self.codes)).cuda().long(),
                    }
                else:
                    self._torch = {}
            except Exception:
                self._torch = {}
        return self._torch

    def topk(self, query_unit: np.ndarray, k: int):
        """ADC top-k (idx, sim) for one or more full-precision unit queries [nq, D]."""
        q = np.asarray(query_unit, np.float32).reshape(-1, self.D)
        k = min(int(k), self.N)
        tt = self._try_torch()
        if tt:
            torch = tt["torch"]
            with torch.no_grad():
                lut = torch.from_numpy(
                    np.einsum("qmd,mkd->qmk", q.reshape(-1, self.M, self.ds),
                              self.codebook).astype(np.float32)).cuda()  # [nq,M,256]
                sc = torch.zeros((q.shape[0], self.N), device="cuda")
                codes = tt["codes"]                                       # [N,M]
                for m in range(self.M):
                    sc += lut[:, m, :][:, codes[:, m]]
                sv, si = torch.topk(sc, k, dim=1)
                return si.cpu().numpy(), sv.cpu().numpy()
        # numpy path
        lut = np.einsum("qmd,mkd->qmk", q.reshape(-1, self.M, self.ds),
                        self.codebook).astype(np.float32)                 # [nq,M,256]
        sc = np.zeros((q.shape[0], self.N), np.float32)
        for m in range(self.M):
            sc += lut[:, m, :][:, self.codes[:, m]]
        idx = np.argpartition(-sc, k - 1, axis=1)[:, :k]
        part = np.take_along_axis(sc, idx, 1)
        o = np.argsort(-part, axis=1)
        return np.take_along_axis(idx, o, 1), np.take_along_axis(part, o, 1)


def nn_neighbors(pq: PqIndex, query_unit, query_item_key: str):
    """Self/alias-excluded top-MAXK neighbor (idx, sim) for one query (training semantics:
    drop exact-key self and any sim >= 1-1e-6 alias; deep buffer for PQ saturation)."""
    nreq = min(pq.N, MAXK + 1 + NN_SEARCH_BUFFER)
    idx, sim = pq.topk(query_unit, nreq)
    idx, sim = idx[0], sim[0]
    thr = 1.0 - NN_ALIAS_EPS
    keep = (sim < thr) & (pq.item_keys[idx] != query_item_key)
    return idx[keep][:MAXK], sim[keep][:MAXK]


def nn_geometry_cols(sim: np.ndarray) -> np.ndarray:
    """geo__local_density / dist_gap_1_to_K / lid_estimate from kept sims (derive_nn)."""
    s = np.asarray(sim, dtype=float)
    if s.size == 0:
        return np.zeros(3, np.float32)
    local_density = float(s.mean())
    gap = float(s[0] - s[-1])
    d = np.clip(1.0 - s, 1e-9, None)
    lid = 0.0
    if d.size > 1 and d[-1] > 0:
        log_ratio = float(np.mean(np.log(d[-1] / d[:-1])))
        if log_ratio > 0:
            lid = 1.0 / log_ratio
    return np.array([local_density, gap, lid], np.float32)


def _nanslope(values, ks) -> float:
    x = np.log2(np.asarray(ks, dtype=float))
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return 0.0
    xx = x[ok] - x[ok].mean()
    yy = y[ok] - y[ok].mean()
    denom = float((xx * xx).sum())
    return float((xx * yy).sum() / denom) if denom > 0 else 0.0


def nn_label_cols(neigh_idx, passrate: CsrPassrate, s_row: int):
    """nn_label_derivatives(13) + counts_subject(1) for ONE (subject,item) given the
    item's kept neighbor index list (derive_nn aggregation, verbatim semantics)."""
    n = len(neigh_idx)
    labels = (passrate.gather_pairs(np.full(n, s_row, np.int64),
                                    np.asarray(neigh_idx, np.int64))
              if n else np.zeros(0))
    means, covs = [], []
    for k in KS:
        lab_k = labels[:k]
        obs = np.isfinite(lab_k)
        means.append(float(np.nanmean(lab_k)) if obs.any() else 0.0)
        covs.append(float(obs.mean()) if len(lab_k) else 0.0)
    obs_all = np.isfinite(labels)
    passed = labels[obs_all]
    q50 = float(np.median(passed)) if passed.size else 0.0
    iqr = float(np.subtract(*np.percentile(passed, [75, 25]))) if passed.size else 0.0
    frac_pass = float((passed > 0.5).mean()) if passed.size else 0.0
    nn13 = np.array(means + covs + [_nanslope(means, KS), _nanslope(covs, KS),
                                    q50, iqr, frac_pass], np.float32)
    return nn13, np.array([float(obs_all.sum())], np.float32)


# ---------------------------------------------------------------------------------
# members (vendored src/mlp_member.py forward + src/tree_numpy.py forest)
# ---------------------------------------------------------------------------------
def _sigmoid_stable(z):
    z = np.asarray(z, np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class MlpMember:
    """Reloadable gated-GLU MLP member (matches MlpMemberState.save layout)."""

    def __init__(self, member_dir):
        d = Path(member_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with np.load(d / "weights.npz", allow_pickle=False) as z:
            self.w = {k: z[k].astype(np.float32) for k in z.files}
        self.n_subjects = int(meta["n_subjects"])
        self.subj_emb_dim = int(meta["subj_emb_dim"])
        self.use_item_emb = bool(meta["use_item_emb"])
        self.item_emb_dim = int(meta["item_emb_dim"])
        self.dense_dim = int(meta["dense_dim"])
        self.dense_feature_names = list(meta["dense_feature_names"])
        self.head_b = float(self.w["head_b"]) if "head_b" in self.w else 0.0

    def predict(self, subject_ids=None, item_emb=None, dense_X=None) -> np.ndarray:
        parts = []
        n = None
        if self.subj_emb_dim > 0:
            s = np.asarray(subject_ids, dtype=np.int64).reshape(-1)
            n = s.shape[0]
            unk = self.n_subjects
            s = np.where((s >= 0) & (s < self.n_subjects), s, unk)
            parts.append(self.w["subj_emb"][s])
        if self.use_item_emb:
            e = np.asarray(item_emb, np.float32).reshape(-1, self.item_emb_dim)
            n = e.shape[0] if n is None else n
            parts.append(e)
        if self.dense_dim > 0:
            m = np.asarray(dense_X, np.float32).reshape(-1, self.dense_dim)
            n = m.shape[0] if n is None else n
            mz = (m - self.w["dense_mean"]) / self.w["dense_std"]
            parts.append(np.where(np.isfinite(mz), mz, 0.0).astype(np.float32))
        x = np.concatenate(parts, axis=1).astype(np.float32)
        w = self.w
        h1 = (x @ w["l1_value_W"] + w["l1_value_b"]) * _sigmoid_stable(
            x @ w["l1_gate_W"] + w["l1_gate_b"]).astype(np.float32)
        h2 = (h1 @ w["l2_value_W"] + w["l2_value_b"]) * _sigmoid_stable(
            h1 @ w["l2_gate_W"] + w["l2_gate_b"]).astype(np.float32)
        z = (h2 @ w["head_W"]).reshape(-1) + self.head_b
        return np.clip(_sigmoid_stable(z), EPS, 1 - EPS)


def load_forest(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def forest_predict(X: np.ndarray, arrs: dict) -> np.ndarray:
    X = np.asarray(X, np.float32).reshape(1, -1) if X.ndim == 1 else np.asarray(X, np.float32)
    feat, thr = arrs["feat"], arrs["thr"]
    left, right = arrs["left"], arrs["right"]
    leaf, isleaf, dleft = arrs["leaf"], arrs["isleaf"], arrs["dleft"]
    off = arrs["offsets"]
    NT = int(arrs["num_tree"])
    base = float(arrs["base_score"])
    n = X.shape[0]
    acc = np.zeros(n, np.float64)
    for ti in range(NT):
        s, e = int(off[ti]), int(off[ti + 1])
        f_, t_, l_, r_ = feat[s:e], thr[s:e], left[s:e], right[s:e]
        il_, dl_, lv_ = isleaf[s:e], dleft[s:e], leaf[s:e]
        node = np.zeros(n, np.int32)
        active = ~il_[node]
        while active.any():
            idx = np.nonzero(active)[0]
            nd = node[idx]
            xv = X[idx, f_[nd]]
            go_left = (xv <= t_[nd]) | (np.isnan(xv) & dl_[nd])
            node[idx] = np.where(go_left, l_[nd], r_[nd])
            active = ~il_[node]
        acc += lv_[node]
    return np.clip(base + acc / NT, EPS, 1 - EPS)


# ---------------------------------------------------------------------------------
# FamilyRuntime: artifacts + caches -> per-(subject,item) family probability
# ---------------------------------------------------------------------------------
class FamilyRuntime:
    """Owns one family's artifacts and computes its L2 (within-family) probability.

    Per-ITEM state (embedding, geometry cols, neighbor lists, fine cluster) is cached by
    item_key; per-(subject,item) label aggregations are cheap numpy on top.
    """

    def __init__(self, fam_dir, shared):
        fam_dir = Path(fam_dir)
        self.meta = json.loads((fam_dir / "artifacts" / "runtime_meta.json")
                               .read_text(encoding="utf-8"))
        art = fam_dir / "artifacts"
        self.shared = shared      # dict: passrate, subj_vocab, n_subjects
        with np.load(art / "centroids.npz", allow_pickle=False) as z:
            self.fine, self.coarse = z["fine"].astype(np.float32), z["coarse"].astype(np.float32)
        with np.load(art / "cluster_aux.npz", allow_pickle=False) as z:
            self.item_to_cluster = z["item_to_cluster"].astype(np.int64)  # passrate col order
            self.cluster_sizes = z["sizes"].astype(np.int64)
            self.sqd_scale = float(z["sqd_scale"])
            # PQ index rows are in the family's embedding-parquet order; passrate columns
            # are in the shared canonical item order — remap neighbors before label gathers
            self.pq_to_col = z["pq_to_passrate_col"].astype(np.int64)
        self.pq = PqIndex(art / "pqidx.npz")
        with np.load(art / "pca_item_emb.npz", allow_pickle=False) as z:
            self.pca_components = z["components"].astype(np.float32)
            self.pca_mean = z["mean"].astype(np.float32)
        self.forest = load_forest(art / "etbig_forest.npz")
        self.mlp_members = {name: MlpMember(art / "mlp" / name)
                            for name in self.meta["l1_members"]}
        self.l1_weights = [float(w) for w in self.meta["l1_weights"]]
        self.l1_bias = float(self.meta["l1_bias"])
        self.l2_weights = self.meta["l2_weights"]        # {"mlp_L1": w, "etbig": w}
        self.l2_bias = float(self.meta["l2_bias"])
        # per-member dense column indices into the canonical 543 (precomputed at export)
        self.member_dense_cols = {k: np.asarray(v, np.int64)
                                  for k, v in self.meta["member_dense_cols"].items()}
        # cluster label aggregates over the shipped train passrate (computed once)
        pr = shared["passrate"]
        self.difficulty, self.smean, self.scnt = pr.cluster_aggregates(
            self.item_to_cluster, self.fine.shape[0])
        self._item_state: dict[str, dict] = {}

    # -- item-level (cached) ---------------------------------------------------------
    def item_state(self, item_key: str, raw_emb: np.ndarray) -> dict:
        st = self._item_state.get(item_key)
        if st is not None:
            return st
        e = np.asarray(raw_emb, np.float32).reshape(1, -1)
        eu = unit_rows(e)
        cd, geo, oh, q_fine, resp = cluster_geometry_block(
            eu, self.fine, self.coarse, self.cluster_sizes, self.sqd_scale)
        n_idx, n_sim = nn_neighbors(self.pq, eu, item_key)
        nn_geo = nn_geometry_cols(n_sim)
        pca = ((e - self.pca_mean) @ self.pca_components.T).astype(np.float32)
        st = {"emb": e[0], "geom525": np.concatenate(
                  [cd[0], geo[0], nn_geo, oh[0]]).astype(np.float32),
              "q_fine": int(q_fine[0]), "resp": resp[0],
              "neigh_idx": n_idx, "pca": pca[0]}
        self._item_state[item_key] = st
        return st

    # -- (subject,item) --------------------------------------------------------------
    def predict_pair(self, subject_key: str, item_key: str, raw_emb: np.ndarray) -> float:
        st = self.item_state(item_key, raw_emb)
        s_id = self.shared["subj_vocab"].get(subject_key, -1)
        s_row = self.shared["passrate_row"].get(subject_key, -1)
        neigh_cols = self.pq_to_col[st["neigh_idx"]] if len(st["neigh_idx"]) else st["neigh_idx"]
        nn13, cnt1 = nn_label_cols(neigh_cols, self.shared["passrate"], s_row)
        pr1, subj3 = cluster_label_cols(st["q_fine"], st["resp"], s_row,
                                        self.difficulty, self.smean, self.scnt)
        dense543 = np.concatenate([st["geom525"], nn13, pr1, subj3, cnt1])
        dense543 = np.where(np.isfinite(dense543), dense543, 0.0).astype(np.float32)

        # L1: pruned mlp LOO members -> linear blend
        l1_probs = []
        for name in self.meta["l1_members"]:
            m = self.mlp_members[name]
            dsel = dense543[self.member_dense_cols[name]] if m.dense_dim > 0 else None
            p = m.predict(
                subject_ids=np.array([s_id]) if m.subj_emb_dim > 0 else None,
                item_emb=st["emb"] if m.use_item_emb else None,
                dense_X=dsel)
            l1_probs.append(float(p[0]))
        p_mlp = linear_blend(l1_probs, self.l1_weights, self.l1_bias)

        # etbig on dense543 + pca
        x_tree = np.concatenate([dense543, st["pca"]]).astype(np.float32)
        p_et = float(forest_predict(x_tree, self.forest)[0])

        # L2: family blend
        return linear_blend([p_mlp, p_et],
                            [self.l2_weights["mlp_L1"], self.l2_weights["etbig"]],
                            self.l2_bias)
