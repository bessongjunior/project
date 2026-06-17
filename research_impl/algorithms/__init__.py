# -*- coding: utf-8 -*-
"""
Map-aware spatio-temporal models for delivery route & ETA prediction.

All models consume the OSM road-network topology produced by
``research_impl/extraction/map_data.py`` (graph -> Laplacian/adjacency tensors).
"""

# Graph cores
from .st_gcn import STGCNLayer, STGCN
from .t_gcn import GCN, TGCN

# Map-aware model wrappers
from .graph2route import MapAwareGraph2Route
from .m2g4rtp import MapAwareM2G4RTP
from .drl4route import MapAwareDRL4Route
from .fdnet import MapAwareFDNet

__all__ = [
    # cores
    "STGCNLayer",
    "STGCN",
    "GCN",
    "TGCN",
    # wrappers
    "MapAwareGraph2Route",
    "MapAwareM2G4RTP",
    "MapAwareDRL4Route",
    "MapAwareFDNet",
]
