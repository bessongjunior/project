# Map-Based Graph Algorithms (ST-GCN, T-GCN & Routing)

This directory contains the core spatio-temporal and routing models that utilize real-world road network topology (OSMNx) for delivery optimization.

## 1. Spatio-Temporal Baselines
- **ST-GCN (`st_gcn.py`):** Uses Chebyshev Polynomials and 1D Causal Convolutions for urban dynamic modeling.
- **T-GCN (`t_gcn.py`):** Combines GCN with GRU for ETA and traffic flow prediction.

## 2. Advanced Map-Aware Routing Models
These are the "Best-in-Class" models adapted to consume physical road network graphs:

- **Graph2Route (`graph2route.py`):** Captures spatial correlations through the physical road graph while maintaining personalized courier embeddings.
- **FDNet (`fdnet.py`):** A coupled route/time predictor that uses Wide & Deep features (including road metadata) for simultaneous sequence and ETA optimization.
- **M2G4RTP (`m2g4rtp.py`):** Uses Graph Attention (GAT) to cluster deliveries into Map-Aware neighborhoods (AOIs) for hierarchical routing.
- **DRL4Route (`drl4route.py`):** A Reinforcement Learning agent that learns optimal navigation policies across the OSM road network.

## 3. Grading Baselines (`baseline/`)
Reference routers used by `evaluation/eval.py` to assign the A-D grade
(evaluation.md §3); not deep models:
- **Distance-Greedy (`baselines.py`):** nearest-unvisited-stop heuristic (D gate).
- **Time-Greedy (`baselines.py`):** earliest-accept-time-first ordering.
- **OR-Tools (`baselines.py`):** exact-ish TSP tour (A gate, optional dep).
- **LightGBM (`lightgbm_baseline.py`):** learned visit-rank regressor (B gate, optional dep).

## 4. Training Strategy
These models are trained using the `project/data/` tensors which map delivery tasks to actual OSM nodes.

### **Key Parameters**
- **$K$ (Chebyshev Order):** Defines the spatial "receptive field" (how many street-hops to consider).
- **Hidden Channels:** Typically 32, 64, or 128 depending on city complexity.
- **Learning Rate:** Optimized for delivery-time minimization (MAE).
