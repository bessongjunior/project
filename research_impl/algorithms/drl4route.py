# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .t_gcn import GCN


class MapAwareDRL4Route(nn.Module):
    """
    DRL4Route adapted for physical OSM road networks.

    An actor-critic agent whose state is the road graph: a GCN encodes the
    current node features over the OSM adjacency, the actor outputs a policy
    over the next stop (masking unreachable nodes) and the critic estimates the
    state value.
    """

    def __init__(self, n_features, n_hidden):
        super(MapAwareDRL4Route, self).__init__()
        self.n_hidden = n_hidden

        # Graph encoder shared by actor and critic.
        self.encoder = GCN(n_features, n_hidden)

        # Policy network (actor): per-node logit -> next-stop distribution.
        self.actor_head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

        # Value network (critic): scalar value of the graph state.
        self.critic_head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x, adj, mask=None):
        """
        x: (B, N, F) node features (courier position + remaining tasks on OSM)
        adj: (N, N) or (B, N, N) OSM adjacency
        mask: (B, N) bool, True = node unavailable
        Returns: (probs (B, N), value (B, 1))
        """
        h = self.encoder(x, adj)                  # (B, N, H)

        logits = self.actor_head(h).squeeze(-1)   # (B, N)
        if mask is not None:
            logits = logits.masked_fill(mask.bool(), float('-inf'))
        probs = F.softmax(logits, dim=-1)         # policy over next node

        value = self.critic_head(h.mean(dim=1))   # (B, 1)
        return probs, value
