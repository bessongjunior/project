# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class STGCNLayer(nn.Module):
    """
    Spatio-Temporal GCN Layer (ChebNet spatial conv + 1D temporal conv).

    The spatial step uses the stable Chebyshev recurrence applied directly to
    the signal (T_k(L) @ x), which is O(K) matmuls instead of the exponential
    recursion of the previous placeholder.
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
        """
        x: (batch_size, num_nodes, in_channels)
        L: Scaled normalized Laplacian (num_nodes, num_nodes) or (B, N, N)
        """
        # --- Spatial: Chebyshev graph convolution ---
        # Tx_0 = x ; Tx_1 = L x ; Tx_k = 2 L Tx_{k-1} - Tx_{k-2}
        Tx_0 = x
        out = torch.matmul(Tx_0, self.theta[0])
        if self.K > 1:
            Tx_1 = torch.matmul(L, x)
            out = out + torch.matmul(Tx_1, self.theta[1])
            for k in range(2, self.K):
                Tx_2 = 2 * torch.matmul(L, Tx_1) - Tx_0
                out = out + torch.matmul(Tx_2, self.theta[k])
                Tx_0, Tx_1 = Tx_1, Tx_2

        # --- Temporal: 1D conv over the node sequence ---
        out = out.permute(0, 2, 1)  # (B, C, N)
        out = self.temporal_conv(out)
        out = out.permute(0, 2, 1)  # (B, N, C)
        return F.relu(out)


class STGCN(nn.Module):
    def __init__(self, n_nodes, n_features, n_hidden, n_output, K=3):
        super(STGCN, self).__init__()
        self.st_layer1 = STGCNLayer(n_features, n_hidden, K)
        self.st_layer2 = STGCNLayer(n_hidden, n_hidden, K)
        self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, x, L):
        x = self.st_layer1(x, L)
        x = self.st_layer2(x, L)
        return self.fc(x)
