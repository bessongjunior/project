# -*- coding: utf-8 -*-
"""
Tensor builder (pipeline stage 3, runs after map-matching).

Turns the map-matched feature table (package_feature_mapped.csv / .parquet) into
per-trajectory training tensors consumed by research_impl/algorithms and graded
by pre_processing/utils.Metric.

One sample == one courier-day trajectory of N stops (N_min <= N <= N_max), padded
to N_max. Sample fields (stacked over samples and saved as a dict .npy):

  V           (S, N_max, F)   node features per stop
  A           (S, N_max, N_max) stop-level adjacency (Gaussian kernel on distance)
  L           (S, N_max, N_max) scaled normalized Laplacian of A (for ST-GCN/ChebNet)
  label       (S, N_max)       ground-truth visit order as input-node indices (-1 pad)
  mask        (S, N_max)       True = padded / unavailable stop
  length      (S,)             real N per sample
  courier_id  (S,)             courier index (for the Graph2Route embedding)
  eta_label   (S, N_max)       per-leg travel time (minutes) to each stop (0 pad)
  coords      (S, N_max, 2)    raw lat/lng in input order (for LSD in metres)
  is_rush     (S,)             1 if the trajectory falls in a peak-traffic window

Numpy-only so it can build tensors without torch installed; training wraps these
arrays in a torch Dataset (see research_impl/train.py). Heavy ingestion is done
upstream in Spark (dataset/dataset.py); this stage is small per-trajectory numpy.
"""
import os
import argparse

import numpy as np
import pandas as pd

from research_impl.pre_processing.utils import ws, dir_check, scaled_laplacian, normalize_adj, CITIES

FEATURE_COLS = [
    'lat', 'lng', 'accept_time_minute', 'expect_finish_time_minute',
    'time_to_last_package', 'dis_to_last_package', 'true_road_distance', 'aoi_type',
]


def _haversine_matrix(lat, lng):
    """Pairwise great-circle distance matrix (metres) for stop coordinates."""
    R = 6371000.0
    lat = np.radians(lat)[:, None]
    lng = np.radians(lng)[:, None]
    dlat = lat - lat.T
    dlng = lng - lng.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlng / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _in_peak(minute, windows=((420, 540), (1020, 1140))):
    """Rush-hour test: minute-of-day in morning (7-9) or evening (17-19) peak."""
    return any(lo <= minute <= hi for lo, hi in windows)


def _adjacency(dist, k=8, eps=1e-6):
    """k-NN Gaussian-kernel adjacency from a distance matrix."""
    n = dist.shape[0]
    sigma = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
    A = np.exp(-(dist ** 2) / (2 * sigma ** 2 + eps))
    np.fill_diagonal(A, 0.0)
    if n > k + 1:                                    # keep only k nearest per row
        for i in range(n):
            thresh = np.sort(A[i])[-k]
            A[i, A[i] < thresh] = 0.0
    A = np.maximum(A, A.T)                            # symmetrize
    return A


class DeliveryDataset:
    def __init__(self, params):
        self.p = params
        self.N_min, self.N_max = params['len_range']
        self.F = 10  # engineered features (see _features): cyclical time encoding

    # -- per-trajectory feature construction ------------------------------
    def _features(self, traj):
        """traj: DataFrame of one courier-day, sorted by finish_time_minute.
        Builds 10 engineered features (cyclical time encoding) per stop."""
        N = len(traj)
        for c in FEATURE_COLS:
            if c not in traj.columns:
                traj[c] = 0.0
        traj = traj.copy()

        lat = traj['lat'].to_numpy(dtype=np.float64)
        lng = traj['lng'].to_numpy(dtype=np.float64)
        acc = traj['accept_time_minute'].to_numpy(dtype=np.float64)
        exp = traj['expect_finish_time_minute'].to_numpy(dtype=np.float64)
        ttl = traj['time_to_last_package'].to_numpy(dtype=np.float64)
        dis = traj['dis_to_last_package'].to_numpy(dtype=np.float64)
        road = traj['true_road_distance'].to_numpy(dtype=np.float64)
        aoi = pd.factorize(traj['aoi_type'])[0].astype(np.float64)

        tp = 2.0 * np.pi
        feats = np.stack([
            lat - lat.mean(),                       # centred latitude
            lng - lng.mean(),                       # centred longitude
            np.sin(tp * acc / 1440.0),              # accept time (cyclical)
            np.cos(tp * acc / 1440.0),
            np.sin(tp * exp / 1440.0),              # expected finish (cyclical)
            np.cos(tp * exp / 1440.0),
            ttl / 60.0,                             # time-to-last (hours)
            dis / 1000.0,                           # geodesic dist-to-last (km)
            road / 1000.0,                          # road dist-to-last (km)
            aoi,                                    # AOI category code
        ], axis=1)                                  # (N, 10)

        dist = _haversine_matrix(lat, lng)
        A = _adjacency(dist, k=self.p.get('knn', 8))

        # Trajectory is sorted by completion time; shuffle input order so the
        # model must recover the sequence (no leakage).
        perm = np.random.permutation(N)
        V = feats[perm]
        A = A[np.ix_(perm, perm)]
        visit_seq = np.argsort(perm)                # input-node pos of t-th visited stop
        eta = np.cumsum(ttl)                        # cumulative trip time (min), visit order
        coords = np.stack([lat, lng], axis=1)[perm]  # raw lat/lng in input order (LSD)
        return V, A, visit_seq, eta, coords

    # -- pad + stack ------------------------------------------------------
    def build(self, df):
        Nmax, F = self.N_max, self.F
        samples = {k: [] for k in
                   ['V', 'A', 'L', 'label', 'mask', 'length', 'courier_id', 'eta_label',
                    'coords', 'is_rush']}

        couriers = {c: i for i, c in enumerate(sorted(df['courier_id'].unique()))}
        group_keys = ['courier_id', 'ds'] if 'ds' in df.columns else ['courier_id']

        for _, traj in df.groupby(group_keys):
            traj = traj.sort_values('finish_time_minute')
            N = len(traj)
            if N < self.N_min or N > Nmax:
                continue
            V, A, visit_seq, eta, coords = self._features(traj)
            L = scaled_laplacian(A)
            A = normalize_adj(A)

            Vp = np.zeros((Nmax, F)); Vp[:N] = V
            Ap = np.zeros((Nmax, Nmax)); Ap[:N, :N] = A
            Lp = np.zeros((Nmax, Nmax)); Lp[:N, :N] = L
            lab = np.full(Nmax, -1, dtype=np.int64); lab[:N] = visit_seq
            mask = np.ones(Nmax, dtype=bool); mask[:N] = False
            etap = np.zeros(Nmax); etap[:N] = eta
            cp = np.zeros((Nmax, 2)); cp[:N] = coords
            rush = int(_in_peak(float(traj['finish_time_minute'].to_numpy().mean())))

            samples['V'].append(Vp)
            samples['A'].append(Ap)
            samples['L'].append(Lp)
            samples['label'].append(lab)
            samples['mask'].append(mask)
            samples['length'].append(N)
            samples['courier_id'].append(couriers[traj['courier_id'].iloc[0]])
            samples['eta_label'].append(etap)
            samples['coords'].append(cp)
            samples['is_rush'].append(rush)

        return {k: np.array(v) for k, v in samples.items()}


def _read_mapped(city):
    """Load the map-matched table for a city (parquet preferred, csv fallback)."""
    base = os.path.join(ws, 'research_impl', 'dataset', 'tmp', city)
    pq = os.path.join(base, 'package_feature_mapped.parquet')
    csv = os.path.join(base, 'package_feature_mapped.csv')
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    if os.path.exists(csv):
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"No mapped table for '{city}' in {base}. Run pre_processing/map_matcher.py first.")


def build_city(city, params):
    df = _read_mapped(city)

    # split trajectories by ratio
    keys = df[['courier_id', 'ds']].drop_duplicates() if 'ds' in df.columns \
        else df[['courier_id']].drop_duplicates()
    keys = keys.sample(frac=1.0, random_state=params.get('seed', 2024)).reset_index(drop=True)
    n = len(keys)
    n_tr = int(n * params['train_ratio'])
    n_va = int(n * params['val_ratio'])
    split = {'train': keys.iloc[:n_tr], 'val': keys.iloc[n_tr:n_tr + n_va],
             'test': keys.iloc[n_tr + n_va:]}

    builder = DeliveryDataset(params)
    out_dir = os.path.join(ws, 'research_impl', 'dataset', city)
    for mode, ks in split.items():
        sub = df.merge(ks, on=list(ks.columns), how='inner')
        data = builder.build(sub)
        out = os.path.join(out_dir, f'{mode}.npy')
        dir_check(out)
        np.save(out, data, allow_pickle=True)
        print(f"[+] {city}/{mode}: {len(data['length'])} samples -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Build delivery training tensors")
    parser.add_argument('--cities', nargs='+', default=CITIES)
    parser.add_argument('--len_range', type=int, nargs=2, default=(2, 25))
    parser.add_argument('--knn', type=int, default=8)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    params = vars(args)
    params['len_range'] = tuple(args.len_range)

    for city in args.cities:
        try:
            build_city(city, params)
        except FileNotFoundError as e:
            print(f"[!] Skipping {city}: {e}")


if __name__ == "__main__":
    main()
