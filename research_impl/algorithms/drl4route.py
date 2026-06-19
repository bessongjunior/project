# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .t_gcn import GCN
from .pointer import PointerDecoder


class MapAwareDRL4Route(nn.Module):
    """
    DRL4Route for OSM road networks: a GCN encoder over the road graph feeding an
    autoregressive pointer decoder. (Re-architected from the one-shot actor-critic
    so beam-search decoding is meaningful; the route is now produced sequentially.)
    """

    def __init__(self, n_features, n_hidden):
        super(MapAwareDRL4Route, self).__init__()
        self.enc = GCN(n_features, n_hidden)
        self.proj = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU())
        self.dec = PointerDecoder(n_hidden, n_hidden)

    def encode(self, x, adj):
        return self.proj(self.enc(x, adj))

    def forward(self, x, adj, mask=None, target=None):
        return self.dec(self.encode(x, adj), mask, target)

    @torch.no_grad()
    def beam_decode(self, x, adj, mask=None, beam=5):
        return self.dec.beam_decode(self.encode(x, adj), mask, beam)
