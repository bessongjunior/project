# -*- coding: utf-8 -*-
"""
Map-Matching Bridge  (see map_matcher.md / docs.md)

Connects the raw delivery logs to the map-aware models by:
  Phase A  - loading the city OSM graph and building its graph tensors,
  Phase B  - snapping every delivery point to its nearest OSM node,
  Phase C  - replacing geodesic distance with true shortest-path road distance.

Outputs (per city):
  research_impl/dataset/tmp/<city>/package_feature_mapped.csv
      adds  osm_node_id, true_road_distance  to package_feature.csv
  research_impl/processed/<city>_delivery_graph.npy
      compact tensor view of the road graph:
      {node_ids, node_index, edge_index (2,E), edge_weight (E,)}

Note on graph representation: a full dense city adjacency is too large to keep
in memory, so we persist a compact edge list here. The dense per-trajectory
operators (A / scaled Laplacian L) the GCN/ST-GCN layers consume are built from
the stops of each trajectory in dataset/dataset.py via utils.scaled_laplacian.
"""
import os
import pickle

import numpy as np
import pandas as pd

from research_impl.pre_processing.utils import ws, dir_check


# ---------------------------------------------------------------------------
# Phase A - graph loading & tensor view
# ---------------------------------------------------------------------------

def load_graph(city):
    """Load a pickled OSM graph saved by extraction/map_data.py."""
    path = os.path.join(ws, 'research_impl', 'processed', f'{city}_graph.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"City graph not found: {path}. Run extraction/map_data.py first.")
    with open(path, 'rb') as f:
        return pickle.load(f)


def city_graph_tensors(G):
    """Compact, memory-safe tensor view of the OSM graph.

    Returns a dict with a stable node ordering plus an edge list weighted by
    travel time (falls back to edge length, then 1.0)."""
    node_ids = np.array(list(G.nodes()))
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    src, dst, w = [], [], []
    for u, v, data in G.edges(data=True):
        src.append(node_index[u])
        dst.append(node_index[v])
        w.append(float(data.get('travel_time', data.get('length', 1.0))))

    return {
        'node_ids': node_ids,
        'node_index': node_index,
        'edge_index': np.array([src, dst], dtype=np.int64),
        'edge_weight': np.array(w, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Phase B - snapping
# ---------------------------------------------------------------------------

def snap_points(G, df, lat_col='lat', lon_col='lng'):
    """Snap every (lat, lng) to its nearest OSM node id."""
    import osmnx as ox
    return ox.nearest_nodes(G, X=df[lon_col].values, Y=df[lat_col].values)


# ---------------------------------------------------------------------------
# Phase C - true road distance
# ---------------------------------------------------------------------------

def _road_distance(G, u, v, cache, weight='length'):
    """Shortest-path distance (metres) between two OSM nodes, with caching."""
    import networkx as nx
    if u == v:
        return 0.0
    key = (u, v)
    if key in cache:
        return cache[key]
    try:
        d = nx.shortest_path_length(G, u, v, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        d = np.nan
    cache[key] = d
    return d


def add_true_road_distance(G, df, node_col='osm_node_id'):
    """Add `true_road_distance`: shortest road distance from each stop to the
    previous stop in the same courier/day trajectory (else 0 for the first)."""
    cache = {}
    nodes = df[node_col].values
    keys = df[['courier_id', 'ds']].values if {'courier_id', 'ds'}.issubset(df.columns) \
        else np.zeros((len(df), 2))

    out = np.zeros(len(df), dtype=np.float64)
    for i in range(len(df)):
        if i == 0 or (keys[i] != keys[i - 1]).any():
            out[i] = 0.0
        else:
            out[i] = _road_distance(G, nodes[i - 1], nodes[i], cache)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _read_features(tmp_dir):
    """Load the per-city feature table written by either ingestion path:
    Spark (package_feature.parquet) preferred, pandas (package_feature.csv) fallback."""
    pq = os.path.join(tmp_dir, 'package_feature.parquet')
    csv = os.path.join(tmp_dir, 'package_feature.csv')
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    if os.path.exists(csv):
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"No package_feature.(parquet|csv) in {tmp_dir}. "
        f"Run the Spark ingestion (dataset/data.py) or pre_processing/preprocess.py first.")


def run(city, with_road_distance=True):
    """Map-match one city end to end."""
    tmp_dir = os.path.join(ws, 'research_impl', 'dataset', 'tmp', city)

    print(f"[*] Map-matching '{city}' ...")
    G = load_graph(city)
    df = _read_features(tmp_dir)

    # Phase B: snap
    df['osm_node_id'] = snap_points(G, df)

    # Phase C: true road distance
    if with_road_distance:
        df['true_road_distance'] = add_true_road_distance(G, df)
    else:
        df['true_road_distance'] = df.get('dis_to_last_package', 0)

    fout = os.path.join(tmp_dir, 'package_feature_mapped.csv')
    dir_check(fout)
    df.to_csv(fout, index=False)
    print(f"[+] Wrote {fout}  ({len(df)} rows)")

    # Phase A: persist compact graph tensors
    tensors = city_graph_tensors(G)
    gpath = os.path.join(ws, 'research_impl', 'processed', f'{city}_delivery_graph.npy')
    dir_check(gpath)
    np.save(gpath, tensors, allow_pickle=True)
    print(f"[+] Wrote {gpath}  ({len(tensors['node_ids'])} nodes, "
          f"{tensors['edge_index'].shape[1]} edges)")
    return df, tensors


if __name__ == "__main__":
    for city in ['chongqing', 'shanghai', 'jilin', 'yantai', 'hangzhou']:
        try:
            run(city)
        except FileNotFoundError as e:
            print(f"[!] Skipping {city}: {e}")
