# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """
    Temporal Graph Convolutional Network (GCN + GRU).

    Consumes a sequence of road-network snapshots and predicts a per-node
    output (e.g. ETA / travel time) at the final step.
    """

    def __init__(self, n_nodes, n_features, n_hidden, n_output):
        super(TGCN, self).__init__()
        self.n_hidden = n_hidden
        self.gcn = GCN(n_features, n_hidden)
        self.gru = nn.GRU(n_hidden, n_hidden, batch_first=True)
        self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, x, A):
        """
        x: (B, T, N, F) sequence of snapshots, or (B, N, F) for a single step.
        A: Adjacency matrix (N, N) or (B, N, N)
        Returns: (B, N, n_output)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, N, F)
        B, T, N, _ = x.shape

        # Spatial convolution at every timestep.
        spatial = []
        for t in range(T):
            spatial.append(self.gcn(x[:, t], A))  # (B, N, H)
        h = torch.stack(spatial, dim=1)           # (B, T, N, H)

        # Temporal modelling per node: GRU over the T axis.
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, self.n_hidden)  # (B*N, T, H)
        out, _ = self.gru(h)
        last = out[:, -1, :].reshape(B, N, self.n_hidden)           # (B, N, H)
        return self.fc(last)
