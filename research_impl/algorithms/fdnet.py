# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class MapAwareFDNet(nn.Module):
    """
    FDNet adapted for physical OSM road networks.

    Couples route prediction with ETA estimation: a shared feature encoder
    feeds an LSTM pointer decoder that emits the delivery sequence, and at every
    decode step a Wide & Deep regressor predicts the travel time to the chosen
    stop from the decoder state and the road-metadata features.
    """

    def __init__(self, n_features, n_hidden):
        super(MapAwareFDNet, self).__init__()
        self.n_hidden = n_hidden

        # Shared feature encoder (time windows, road speeds, distances).
        self.feature_mlp = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        # Route predictor (pointer mechanism).
        self.route_lstm = nn.LSTMCell(n_hidden, n_hidden)
        self.route_pointer = nn.Linear(n_hidden, n_hidden)

        # ETA predictor (Wide & Deep): decoder state + chosen-node feature.
        self.eta_regressor = nn.Sequential(
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x, mask=None):
        """
        x: (B, N, F) features mapped to OSM nodes
        mask: (B, N) bool, True = stop unavailable
        Returns: (route_logits (B, N, N), eta (B, N))
                 route_logits[:, t] is the distribution over the t-th stop;
                 eta[:, t] is the predicted travel time to that stop.
        """
        B, N, _ = x.shape
        device = x.device

        h = F.relu(self.feature_mlp(x))           # (B, N, H)
        ref = self.route_pointer(h)               # (B, N, H)

        if mask is None:
            m = torch.zeros(B, N, dtype=torch.bool, device=device)
        else:
            m = mask.clone().bool()

        dec_h = h.mean(dim=1)                      # (B, H)
        dec_c = torch.zeros_like(dec_h)
        dec_input = h.mean(dim=1)

        route_logits, etas = [], []
        for _ in range(N):
            dec_h, dec_c = self.route_lstm(dec_input, (dec_h, dec_c))

            # Dot-product pointer over remaining stops.
            scores = torch.matmul(ref, dec_h.unsqueeze(-1)).squeeze(-1)  # (B, N)
            scores = scores.masked_fill(m, float('-inf'))
            route_logits.append(scores)

            choice = scores.argmax(dim=-1)                                # (B,)
            m = m.scatter(1, choice.unsqueeze(1), True)
            chosen = h[torch.arange(B, device=device), choice]           # (B, H)
            dec_input = chosen

            # Coupled ETA for the chosen stop.
            eta = self.eta_regressor(torch.cat([dec_h, chosen], dim=-1)).squeeze(-1)  # (B,)
            etas.append(eta)

        return torch.stack(route_logits, dim=1), torch.stack(etas, dim=1)
