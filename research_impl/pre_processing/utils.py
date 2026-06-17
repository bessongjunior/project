# -*- coding: utf-8 -*-
"""
Shared utilities for the delivery (map_data/) pipeline.

Single consolidated module, co-located with the pre-processing bridge. Contains:
  1. Workspace / IO helpers          (ws, dir_check, dict_merge, save2file_meta, ...)
  2. Graph operators                 (add_self_loops, normalize_adj, scaled_laplacian)
  3. Evaluation metrics              (Metric: HR@K, KRC, LSD, ED, MAE/RMSE/MAPE/ACC@T)

Kept dependency-light: only numpy is required at import time; torch is imported
lazily inside the functions that need it. See evaluation.md and docs.md.
"""
import os
import csv
import time
import random
import numpy as np


# ===========================================================================
# 1. Workspace / IO helpers
# ===========================================================================

def get_workspace():
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)   # research_impl/pre_processing/
    file = os.path.dirname(file)       # research_impl/
    file = os.path.dirname(file)       # project root
    return file


ws = get_workspace()


# ---------------------------------------------------------------------------
# City registry — single source of truth for the cities in scope.
# Keeps every stage (extraction / preprocess / map_matcher / dataset) separate
# but consistent. Add a city here once and the whole pipeline picks it up.
# Codes match the LaDe delivery sub-datasets: delivery_<code>.csv
# ---------------------------------------------------------------------------
CITIES = ['shanghai', 'hangzhou', 'chongqing', 'jilin', 'yantai']

CITY_CODE = {            # city key -> raw-file code (delivery_<code>.csv)
    'shanghai': 'sh',
    'hangzhou': 'hz',
    'chongqing': 'cq',
    'jilin': 'jl',
    'yantai': 'yt',
}

CITY_PLACE = {           # city key -> OSMnx place query
    'shanghai': 'Shanghai, China',
    'hangzhou': 'Hangzhou, China',
    'chongqing': 'Chongqing, China',
    'jilin': 'Jilin, China',
    'yantai': 'Yantai, China',
}


def dir_check(path):
    """Ensure the parent directory of `path` exists; return `path`."""
    dir_path = path if os.path.isdir(path) else os.path.split(path)[0]
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return path


def dict_merge(dict_list=None):
    """Shallow-merge a list of dicts into one."""
    out = {}
    for d in (dict_list or []):
        out.update(d)
    return out


def set_seed(seed=2024):
    """Seed python/numpy/torch RNGs for reproducible delivery experiments."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def to_device(batch, device):
    """Recursively move tensors in a dict/list/tuple (or a bare tensor) to `device`."""
    import torch
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device) for v in batch)
    return batch


def save2file_meta(params, file_name, head):
    """Append a row of `params` (selected/ordered by `head`) to a CSV, writing the
    header row once."""
    def timestamp2str(stamp):
        utc_t = int(stamp)
        utc_h = utc_t // 3600
        utc_m = (utc_t // 60) - utc_h * 60
        utc_s = utc_t % 60
        hour = (utc_h + 8) % 24            # CST (UTC+8)
        return f'{hour}:{utc_m}:{utc_s}'

    dir_check(file_name)
    if not os.path.exists(file_name):
        with open(file_name, "w", newline='\n') as f:
            csv.writer(f).writerow(head)
    with open(file_name, "a", newline='\n') as f:
        params['log_time'] = timestamp2str(time.time())
        csv.writer(f).writerow([params.get(k, '') for k in head])


# ===========================================================================
# 2. Graph operators for the map-aware models
#    Numpy in / numpy out; the caller (map_matcher.py / training loop) converts
#    to tensors. Produces the exact operators map_data/algorithms expect.
# ===========================================================================

def add_self_loops(A):
    """A + I. Guarantees every node connects to itself — needed before
    normalization and for edge-aware attention in MapAwareM2G4RTP."""
    A = np.asarray(A, dtype=np.float64)
    return A + np.eye(A.shape[0], dtype=A.dtype)


def normalize_adj(A, add_loops=True):
    """Symmetric normalization  D^{-1/2} A D^{-1/2}  for the GCN family
    (GCN, TGCN, MapAwareM2G4RTP, MapAwareDRL4Route)."""
    A = np.asarray(A, dtype=np.float64)
    if add_loops:
        A = add_self_loops(A)
    deg = A.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = np.power(deg[nz], -0.5)
    D = np.diag(d_inv_sqrt)
    return D @ A @ D


def scaled_laplacian(A, add_loops=True):
    """Scaled normalized Laplacian  L~ = 2L/lambda_max - I  (eigenvalues in
    [-1, 1]), as required by the Chebyshev conv in STGCNLayer /
    MapAwareGraph2Route (see docs.md section 4, open decision #3)."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    L = np.eye(n) - normalize_adj(A, add_loops=add_loops)
    try:
        lam_max = float(np.linalg.eigvalsh(L).max())
    except np.linalg.LinAlgError:
        lam_max = 2.0
    if lam_max == 0:
        lam_max = 2.0
    return (2.0 / lam_max) * L - np.eye(n)


# ===========================================================================
# 3. Evaluation metrics (see evaluation.md)
#    Route/sequence:  HR@K, KRC (Kendall), LSD (location seq distance), ED
#    Travel time/ETA: MAE, RMSE, MAPE, ACC@T
#    Kendall tau, Levenshtein and haversine are implemented locally (no scipy).
# ===========================================================================

def _to_numpy(x):
    """Accept torch tensors, numpy arrays, or python sequences and return a
    numpy array. Lets Metric consume model outputs (torch tensors) directly."""
    if x is None:
        return None
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _haversine(a, b):
    """Great-circle distance in metres between (lat, lng) pairs."""
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))


def _kendall_tau(x, y):
    """Kendall's rank correlation in [-1, 1] (tau-a, O(n^2) — fine for route lengths)."""
    n = len(x)
    if n < 2:
        return 0.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    denom = 0.5 * n * (n - 1)
    return (conc - disc) / denom if denom else 0.0


def _edit_distance(a, b):
    """Levenshtein distance between two sequences."""
    a, b = list(a), list(b)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


class Metric:
    """Accumulating evaluator for delivery route + ETA prediction.

    Usage:
        m = Metric()
        m.update_route(pred_seq, true_seq, scores=..., coords=...)   # per sample
        m.update_eta(pred_times, true_times)                         # per sample
        results = m.summary()                                        # dict
    """

    def __init__(self, k_values=(1, 3, 5), acc_thresholds=(10, 20)):
        self.k_values = tuple(k_values)
        self.acc_thresholds = tuple(acc_thresholds)
        self.reset()

    def reset(self):
        self._route = []
        self._eta_pred = []
        self._eta_true = []

    # ---- route / sequence ------------------------------------------------
    def update_route(self, pred, label, scores=None, coords=None):
        """
        pred:   predicted ordered stop ids, shape (S,)
        label:  ground-truth ordered stop ids, shape (S,)
        scores: optional (S, N) per-step candidate scores (logits) for HR@K;
                if None, HR@K falls back to positional tolerance on `pred`.
        coords: optional (N, 2) array or {id: (lat, lng)} for LSD in metres;
                if None, LSD falls back to a 0/1 positional mismatch.
        """
        pred = _to_numpy(pred).ravel()
        label = _to_numpy(label).ravel()
        if scores is not None:
            scores = _to_numpy(scores)
        if coords is not None and not isinstance(coords, dict):
            coords = _to_numpy(coords)
        S = min(len(pred), len(label))
        pred, label = pred[:S], label[:S]

        rec = {}
        for k in self.k_values:
            rec[f'hr@{k}'] = self._hr_at_k(pred, label, scores, k)
        rec['krc'] = self._krc(pred, label)
        rec['ed'] = _edit_distance(pred, label)
        rec['lsd'] = self._lsd(pred, label, coords)
        self._route.append(rec)

    def _hr_at_k(self, pred, label, scores, k):
        S = len(label)
        if S == 0:
            return 0.0
        if scores is not None:
            scores = np.asarray(scores)
            kk = min(k, scores.shape[1])
            hits = 0
            for t in range(min(S, scores.shape[0])):
                topk = np.argpartition(-scores[t], kk - 1)[:kk]
                if label[t] in topk:
                    hits += 1
            return hits / S
        pos = {stop: i for i, stop in enumerate(pred)}
        hits = sum(1 for i, stop in enumerate(label)
                   if stop in pos and abs(pos[stop] - i) < k)
        return hits / S

    def _krc(self, pred, label):
        pos = {stop: i for i, stop in enumerate(pred)}
        pred_rank = [pos.get(stop, len(pred)) for stop in label]
        true_rank = list(range(len(label)))
        return _kendall_tau(pred_rank, true_rank)

    def _lsd(self, pred, label, coords):
        S = len(label)
        if S == 0:
            return 0.0
        dists = []
        for i in range(S):
            if coords is not None:
                pc, lc = self._coord(coords, pred[i]), self._coord(coords, label[i])
                if pc is not None and lc is not None:
                    dists.append(_haversine(pc, lc))
                    continue
            dists.append(float(pred[i] != label[i]))
        return float(np.mean(dists)) if dists else 0.0

    @staticmethod
    def _coord(coords, idx):
        try:
            if isinstance(coords, dict):
                return coords.get(idx)
            return coords[int(idx)]
        except (IndexError, KeyError, ValueError, TypeError):
            return None

    # ---- ETA -------------------------------------------------------------
    def update_eta(self, pred, true):
        pred = _to_numpy(pred).astype(np.float64).ravel()
        true = _to_numpy(true).astype(np.float64).ravel()
        n = min(len(pred), len(true))
        if n:
            self._eta_pred.append(pred[:n])
            self._eta_true.append(true[:n])

    # ---- summary ---------------------------------------------------------
    def summary(self):
        out = {}
        if self._route:
            for key in self._route[0]:
                out[key] = float(np.mean([r[key] for r in self._route]))
        if self._eta_pred:
            p = np.concatenate(self._eta_pred)
            t = np.concatenate(self._eta_true)
            err = p - t
            out['mae'] = float(np.mean(np.abs(err)))
            out['rmse'] = float(np.sqrt(np.mean(err ** 2)))
            mask = t != 0
            out['mape'] = float(np.mean(np.abs(err[mask] / t[mask])) * 100) if mask.any() else 0.0
            for thr in self.acc_thresholds:
                out[f'acc@{thr}'] = float(np.mean(np.abs(err) <= thr))
        return out

    def __repr__(self):
        return f"Metric({self.summary()})"
