# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .t_gcn import GCN
from .pointer import PointerDecoder


class MapAwareM2G4RTP(nn.Module):
    """
    M2G4RTP for OSM road networks: GCN over the road adjacency + multi-head graph
    attention (AOI neighbourhoods) as the encoder, then an autoregressive pointer
    decoder for the delivery sequence.
    """

    def __init__(self, n_features, n_hidden, n_heads=8):
        super(MapAwareM2G4RTP, self).__init__()
        self.node_proj = nn.Linear(n_features, n_hidden)
        self.gcn = GCN(n_hidden, n_hidden)                 # consumes OSM adjacency
        self.gat = nn.MultiheadAttention(n_hidden, n_heads, batch_first=True)
        self.dec = PointerDecoder(n_hidden, n_hidden)

    def encode(self, x, adj):
        h = F.relu(self.node_proj(x))                      # (B, N, H)
        h = self.gcn(h, adj)                               # spatial propagation
        h_attn, _ = self.gat(h, h, h)                      # AOI neighbourhood attention
        return h_attn

    def forward(self, x, adj, mask=None, target=None):
        return self.dec(self.encode(x, adj), mask, target)

    @torch.no_grad()
    def beam_decode(self, x, adj, mask=None, beam=5):
        return self.dec.beam_decode(self.encode(x, adj), mask, beam)
