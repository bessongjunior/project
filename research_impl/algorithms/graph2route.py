# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .st_gcn import STGCNLayer


class MapAwareGraph2Route(nn.Module):
    """
    Graph2Route adapted for physical OSM road networks.

    ST-GCN layers extract spatial structure from the road graph, a GRU adds
    trajectory memory, a per-courier embedding personalises the encoding, and a
    pointer decoder emits the delivery sequence greedily while masking visited
    / unreachable stops.
    """

    def __init__(self, n_nodes, n_features, n_hidden, n_couriers, d_w=20, K=3):
        super(MapAwareGraph2Route, self).__init__()
        self.n_hidden = n_hidden
        self.d_w = d_w

        # Personalised courier embedding
        self.worker_emb = nn.Embedding(n_couriers, d_w)

        # Map-aware encoder (physical topology) + temporal memory
        self.spatial_encoder = STGCNLayer(n_features, n_hidden, K)
        self.temporal_gru = nn.GRU(n_hidden, n_hidden, batch_first=True)

        # Pointer-based decoder
        self.decoder_rnn = nn.LSTMCell(n_hidden + d_w, n_hidden + d_w)
        self.pointer_query = nn.Linear(n_hidden + d_w, n_hidden)
        self.pointer_ref = nn.Linear(n_hidden, n_hidden)
        self.pointer_v = nn.Linear(n_hidden, 1, bias=False)

    def forward(self, x, L, courier_id, V_reach_mask=None):
        """
        x: (B, N, F) features mapped to OSM nodes
        L: Laplacian from the OSM graph (N, N) or (B, N, N)
        courier_id: (B,)
        V_reach_mask: (B, N) bool, True = stop unavailable (visited/unreachable)
        Returns: (B, N, N) pointer logits (decode step x candidate node)
        """
        B, N, _ = x.shape
        device = x.device

        # 1. Spatial phase (OSM graph)
        h_spatial = self.spatial_encoder(x, L)              # (B, N, H)

        # 2. Temporal phase (trajectory memory)
        h_temporal, _ = self.temporal_gru(h_spatial)        # (B, N, H)

        # 3. Personalised phase
        w_emb = self.worker_emb(courier_id).unsqueeze(1).expand(-1, N, -1)  # (B, N, d_w)
        enc = torch.cat([h_temporal, w_emb], dim=-1)        # (B, N, H+d_w)
        ref = self.pointer_ref(h_temporal)                  # (B, N, H)

        if V_reach_mask is None:
            mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        else:
            mask = V_reach_mask.clone().bool()

        # 4. Greedy pointer decoding
        dec_h = enc.mean(dim=1)                              # (B, H+d_w)
        dec_c = torch.zeros_like(dec_h)
        dec_input = enc.mean(dim=1)

        logits = []
        for _ in range(N):
            dec_h, dec_c = self.decoder_rnn(dec_input, (dec_h, dec_c))
            q = self.pointer_query(dec_h).unsqueeze(1)       # (B, 1, H)
            scores = self.pointer_v(torch.tanh(q + ref)).squeeze(-1)  # (B, N)
            scores = scores.masked_fill(mask, float('-inf'))
            logits.append(scores)

            choice = scores.argmax(dim=-1)                   # (B,)
            mask = mask.scatter(1, choice.unsqueeze(1), True)
            dec_input = enc[torch.arange(B, device=device), choice]  # (B, H+d_w)

        return torch.stack(logits, dim=1)                    # (B, N, N)
