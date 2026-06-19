# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointer import PointerDecoder


class STGCNLayer(nn.Module):
    """
    Spatio-Temporal GCN Layer (ChebNet spatial conv + 1D temporal conv).
    """

    def __init__(self, in_channels, out_channels, K=3):
        super(STGCNLayer, self).__init__()
        self.K = K
        self.theta = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
        self.temporal_conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x, L):
        """x: (B, N, in_channels); L: scaled Laplacian (N, N) or (B, N, N)."""
        Tx_0 = x
        out = torch.matmul(Tx_0, self.theta[0])
        if self.K > 1:
            Tx_1 = torch.matmul(L, x)
            out = out + torch.matmul(Tx_1, self.theta[1])
            for k in range(2, self.K):
                Tx_2 = 2 * torch.matmul(L, Tx_1) - Tx_0
                out = out + torch.matmul(Tx_2, self.theta[k])
                Tx_0, Tx_1 = Tx_1, Tx_2
        out = out.permute(0, 2, 1)
        out = self.temporal_conv(out)
        out = out.permute(0, 2, 1)
        return F.relu(out)


class STGCN(nn.Module):
    """ST-GCN spatial encoder + autoregressive pointer decoder (route prediction)."""

    def __init__(self, n_nodes, n_features, n_hidden, n_output=1, K=3):
        super(STGCN, self).__init__()
        self.l1 = STGCNLayer(n_features, n_hidden, K)
        self.l2 = STGCNLayer(n_hidden, n_hidden, K)
        self.dec = PointerDecoder(n_hidden, n_hidden, predict_eta=True)  # ST-GCN: route + ETA

    def encode(self, x, L):
        return self.l2(self.l1(x, L), L)

    def forward(self, x, L, mask=None, target=None):
        return self.dec(self.encode(x, L), mask, target)

    @torch.no_grad()
    def beam_decode(self, x, L, mask=None, beam=5):
        return self.dec.beam_decode(self.encode(x, L), mask, beam)
