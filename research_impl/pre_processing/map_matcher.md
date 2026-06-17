# Map-Matching Bridge: Data Alignment Strategy

This document outlines the strategy for the `map_matcher` utility, which acts as the technical bridge between raw delivery logs and the map-aware deep learning models.

## 1. Objective
The goal is to transform "Coordinate-Aware" data (raw lat/lng) into "Map-Aware" data (OSM Node IDs). This ensures that routing algorithms respect physical road constraints such as one-way streets, bridges, and speed limits.

## 2. Technical Workflow

### **Phase A: Infrastructure Setup**
- **Road Network Extraction**: Utilize `map_data/map_data.py` to generate or load the high-fidelity OSM road graph (`.pkl`).
- **Graph Transformation**: Calculate the Normalized Laplacian ($L$) and Adjacency Matrix ($A$) from the physical street topology.

### **Phase B: The Snapping Process**
- **Node Mapping**: For every delivery event in the `package_feature.csv`:
    - Perform a spatial search using `osmnx.nearest_nodes` to find the closest intersection/node on the road graph.
    - Store the resulting `osm_node_id` as a primary feature.
- **Trajectory Alignment**: Ensure the sequence of delivery points for a courier is snapped to nodes that are physically reachable in the road network.

### **Phase C: Feature Enrichment (True Distance)**
- **Geodesic vs. Graph**: Replace the standard Euclidean/Geodesic distance ("as the crow flies") with the **Shortest Path Distance** calculated via Dijkstra/A* on the road network.
- **Temporal Alignment**: Estimate base travel times using the `travel_time` metadata extracted from OSM edges.

## 3. Usage & Alignment (Multi-City Support)
The `map_matcher` utility is designed for batch processing. For every city defined in the research scope (e.g., Chongqing, Shanghai, Jilin, Yantai, Hangzhou), it will produce:
1.  `{city}_package_feature_mapped.csv`: Includes `osm_node_id` and `true_road_distance`.
2.  `{city}_delivery_graph.npy`: A tensor representation of the city's specific road network.

### **Model Integration**
The output from this bridge is the mandatory input for the following models:
- `MapAwareGraph2Route`
- `MapAwareFDNet`
- `MapAwareM2G4RTP`
- `MapAwareDRL4Route`

## 4. Evaluation Standards
The success of the map-matching bridge is measured by:
- **Snapping Accuracy**: Percentage of points snapped within 50 meters of their raw coordinate.
- **Graph Connectivity**: Ensuring the resulting trajectory nodes are part of a strongly connected component in the road graph.
