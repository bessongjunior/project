# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointer import PointerDecoder


class GCN(nn.Module):
    """
    Standard GCN layer for spatial feature extraction: relu(A x W).
    """

    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.W = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, A):
        """
        x: (batch_size, num_nodes, in_channels)
        A: Adjacency matrix (num_nodes, num_nodes) or (B, N, N)
        """
        out = self.W(x)             # (B, N, H)
        out = torch.matmul(A, out)  # propagate over edges
        return F.relu(out)


class TGCN(nn.Module):
    """GCN + GRU spatial-temporal encoder + autoregressive pointer decoder."""

    def __init__(self, n_nodes, n_features, n_hidden, n_output=1):
        super(TGCN, self).__init__()
        self.gcn = GCN(n_features, n_hidden)
        self.gru = nn.GRU(n_hidden, n_hidden, batch_first=True)
        self.dec = PointerDecoder(n_hidden, n_hidden, predict_eta=True)  # T-GCN: route + ETA

    def encode(self, x, A):
        h = self.gcn(x, A)          # (B, N, H)
        out, _ = self.gru(h)        # (B, N, H) — recurrent memory over nodes
        return out

    def forward(self, x, A, mask=None, target=None):
        return self.dec(self.encode(x, A), mask, target)

    @torch.no_grad()
    def beam_decode(self, x, A, mask=None, beam=5):
        return self.dec.beam_decode(self.encode(x, A), mask, beam)
