# Pre-Processing & Map-Matching: Status and Bridge Specification

This document records the state of the map-aware delivery pipeline (now under
`research_impl/`), the contract between the map-matching bridge and the models,
and the design decisions taken. It complements `map_matcher.md` (the strategy
doc) with the concrete tensor interface the models expect.

---

## 1. Current State

| Component | Location | Status |
|-----------|----------|--------|
| Ingestion (Spark) | `research_impl/dataset/dataset.py` | **Implemented** — LaDe-D parquet → per-city `package_feature.parquet` (window-based trajectory features) |
| EMR driver | `research_impl/dataset/data.py` | **Implemented** — spark-submit entry point (local or AWS EMR, S3 in/out) |
| Preprocessing (pandas) | `research_impl/pre_processing/preprocess.py` | **Implemented** — small/local alternative to the Spark ingestion |
| Road-graph extractor | `research_impl/extraction/map_data.py` | **Implemented** (OSMnx → `<city>_graph.pkl`) |
| Shared utils + metrics | `research_impl/pre_processing/utils.py` | **Implemented** — IO helpers, city registry, graph ops, torch-capable `Metric` |
| Map-matching bridge | `research_impl/pre_processing/map_matcher.py` | **Implemented** — snap + true road distance + `<city>_delivery_graph.npy` |
| Tensor builder | `research_impl/dataset/tensorize.py` | **Implemented** — mapped table → padded train/val/test tensors |
| Map-aware models | `research_impl/algorithms/*.py` | **Implemented** (real forwards) |
| Baselines | `research_impl/algorithms/baseline/` | **Implemented** — distance-greedy, time-greedy, OR-Tools, LightGBM (for A-D grading) |
| Training harness | `research_impl/train.py` | **Implemented** — trains a model, saves weights + cfg, prints val metrics |
| Evaluation runner | `research_impl/evaluation/eval.py` | **Implemented** — metrics + baselines + A-D grade (evaluation.md §3) |

All Python is **AST-verified but unrun** (no spark/torch/osmnx env yet).

> Scope note: the top-level `algorithm/` folder is the **pickup** reference
> codebase, kept for porting only. The delivery build lives entirely under
> `research_impl/`.

### Data schema (LaDe-D delivery, confirmed)
Columns: `order_id, region_id, city, courier_id, lng, lat, aoi_id, aoi_type,
accept_time, accept_gps_*, delivery_time, delivery_gps_*, ds`. Times are
`"MM-dd HH:mm:ss"`; `ds` is `MMdd`. **No promised time window** → `expect_finish`
defaults to 1440 in both ingestion paths.

### Still needed before it runs end to end
- **venv / packages:** see `requirements.txt`.
- **Raw data:** `research_impl/dataset/csv/delivery_<code>.csv` (have: cq, jl, yt).
- **Remaining (non-blocking):** rush-hour Temporal MAPE gate; scores-based HR@K
  for pointer models.

---

## 2. The Map-Matching Bridge (`map_matcher.py`) — as implemented

Turns coordinate-aware data into map-aware data. Mandatory input producer for
the map-aware models.

### Inputs
- City road graph: `research_impl/processed/<city>_graph.pkl` (from `extraction/map_data.py`).
- Delivery features: `research_impl/dataset/tmp/<city>/package_feature.csv`.

### Steps (Phases A–C from `map_matcher.md`)
- **A — Graph tensors:** compact, memory-safe view of the OSM graph
  (`node_ids`, `node_index`, `edge_index (2,E)`, `edge_weight (E,)`).
- **B — Snapping:** `osmnx.nearest_nodes` → `osm_node_id` per delivery event.
- **C — True distance:** shortest-path road distance to the previous stop
  (`networkx.shortest_path_length`) → `true_road_distance`.

### Outputs
- `research_impl/dataset/tmp/<city>/package_feature_mapped.csv` — adds
  `osm_node_id`, `true_road_distance`.
- `research_impl/processed/<city>_graph.npy` — the compact graph tensors above.

> **Design note (resolves old open-decision #1):** a full *dense* city
> adjacency is too large to persist, so `map_matcher.py` saves a compact edge
> list. The dense per-trajectory operators `A` / scaled `L` that the GCN/ST-GCN
> layers consume are built in `dataset/dataset.py` from the **stops of each
> trajectory** (pairwise distance → Gaussian-kernel kNN adjacency → Laplacian).

---

## 3. Model Tensor Interface

Common contract across `research_impl/algorithms/`, satisfied by the dataset
builder.

### Inputs
- `x` — node features `(B, N, F)`. (`TGCN` also accepts `(B, T, N, F)`.)
- Graph operator:
  - `L` — scaled Laplacian, for the **ST-GCN family** (`STGCN`, `STGCNLayer`,
    `MapAwareGraph2Route`). `(N, N)` or `(B, N, N)`.
  - `adj` — normalized adjacency, for the **GCN family** (`GCN`, `TGCN`,
    `MapAwareM2G4RTP`, `MapAwareDRL4Route`). `(N, N)` or `(B, N, N)`.
- `mask` / `V_reach_mask` — `(B, N)` bool, `True` = stop unavailable.
- `courier_id` — `(B,)` long (Graph2Route only).

### Outputs
| Model | Output | Meaning |
|-------|--------|---------|
| `STGCN` | `(B, N, n_output)` | per-node regression (ETA) |
| `TGCN` | `(B, N, n_output)` | per-node temporal prediction |
| `MapAwareGraph2Route` | `(B, N, N)` | pointer logits: step × candidate |
| `MapAwareFDNet` | `(route_logits (B,N,N), eta (B,N))` | coupled sequence + ETA |
| `MapAwareM2G4RTP` | `(B, N)` | AOI-grounded routing scores |
| `MapAwareDRL4Route` | `(probs (B,N), value (B,1))` | next-stop policy + value |

### Dataset sample fields (`dataset/dataset.py`)
`V (S,N,F)`, `A (S,N,N)`, `L (S,N,N)`, `label (S,N)` visit order (-1 pad),
`mask (S,N)`, `length (S,)`, `courier_id (S,)`, `eta_label (S,N)`.

---

## 4. Resolved Design Decisions

1. **`<city>_graph.npy` layout** — compact dict `{node_ids, node_index,
   edge_index, edge_weight}` (see §2 design note). Per-sample dense `A`/`L`
   built in the dataset.
2. **Self-loops** — `utils.add_self_loops` / `normalize_adj(add_loops=True)`
   add `I` before normalization, so no node is fully isolated.
3. **Chebyshev scaling** — `utils.scaled_laplacian` emits `L̃ = 2L/λ_max − I`
   (eigenvalues in `[-1, 1]`); the dataset stores this as `L`.
4. **Variable `N`** — trajectories are padded to `N_max` (default 25) with a
   `mask` marking padded stops, consumed by the pointer/policy outputs.

---

## 5. Run Order

```
# 0. venv: torch numpy pandas pyarrow huggingface_hub osmnx geopy geohash2 tqdm pyspark

# 1. DOWNLOAD + stage (HuggingFace -> dataset/parquets/ + dataset/csv/)
python -m research_impl.dataset.download               # delivery_<code>.parquet + .csv
#   offline parquet->csv only: python -m research_impl.dataset.download --mode convert

# 2. INGEST (PySpark batch; local or AWS EMR via dataset/data.py)
spark-submit research_impl/dataset/data.py \
    --input research_impl/dataset/parquets --output research_impl/dataset
#   -> dataset/tmp/<city>/package_feature.parquet
#   pandas/CSV alternative: python -m research_impl.pre_processing.preprocess  (reads dataset/csv/)

# 3. EXTRACT road graphs
python -m research_impl.extraction.map_data            # -> processed/<city>_graph.pkl

# 4. MAP-MATCH (snap + true road distance + graph tensors)
python -m research_impl.pre_processing.map_matcher     # -> tmp/<city>/package_feature_mapped.*, processed/<city>_delivery_graph.npy

# 5. TENSORIZE (build per-trajectory training tensors)
python -m research_impl.dataset.tensorize             # -> dataset/<city>/{train,val,test}.npy

# 6. TRAIN
python -m research_impl.train --model graph2route --city chongqing
# then: research_impl/evaluation/eval.py  (uses utils.Metric)
```

> Data staging: `download.py` writes both **parquet** (`dataset/parquets/`, for the
> Spark ingestion) and **CSV** (`dataset/csv/`, for the pandas path). Stage 2
> (`dataset.py`/`data.py`) is the scalable PySpark ingestion; `data.py` is the
> EMR entry point. The numpy tensor builder lives in `dataset/tensorize.py`.
