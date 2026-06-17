# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .t_gcn import GCN


class MapAwareM2G4RTP(nn.Module):
    """
    M2G4RTP adapted for physical OSM road networks.

    A GCN propagates features over the road adjacency, multi-head graph
    attention captures the relative importance of neighbouring stops, and an
    AOI head produces cluster-aware scores used for hierarchical routing.
    """

    def __init__(self, n_features, n_hidden, n_heads=8):
        super(MapAwareM2G4RTP, self).__init__()
        self.n_hidden = n_hidden

        self.node_proj = nn.Linear(n_features, n_hidden)
        self.gcn = GCN(n_hidden, n_hidden)                      # consumes OSM adjacency
        self.gat = nn.MultiheadAttention(n_hidden, n_heads, batch_first=True)
        self.aoi_predictor = nn.Linear(n_hidden, n_hidden)
        self.out_head = nn.Linear(n_hidden, 1)

    def forward(self, x, adj):
        """
        x: (B, N, F) node features
        adj: (N, N) or (B, N, N) OSM adjacency
        Returns: (B, N) routing scores grounded in AOI clusters
        """
        h = F.relu(self.node_proj(x))            # (B, N, H)
        h = self.gcn(h, adj)                      # spatial propagation over edges

        # Multi-head attention captures importance of nearby OSM nodes.
        h_attn, _ = self.gat(h, h, h)            # (B, N, H)

        aoi = torch.tanh(self.aoi_predictor(h_attn))   # AOI cluster representation
        scores = self.out_head(aoi).squeeze(-1)        # (B, N)
        return scores
