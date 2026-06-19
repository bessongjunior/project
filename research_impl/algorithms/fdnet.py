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

    def forward(self, x, mask=None, target=None):
        """
        x: (B, N, F) features mapped to OSM nodes
        mask: (B, N) bool, True = stop unavailable
        target: (B, N) ground-truth visit order; if given, decode with teacher
                forcing (training). If None, decode greedily (inference).
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

        idx = torch.arange(B, device=device)
        route_logits, etas = [], []
        for t in range(N):
            dec_h, dec_c = self.route_lstm(dec_input, (dec_h, dec_c))

            # Dot-product pointer over remaining stops.
            scores = torch.matmul(ref, dec_h.unsqueeze(-1)).squeeze(-1)  # (B, N)
            scores = scores.masked_fill(m, float('-inf'))
            route_logits.append(scores)

            if target is not None:                                       # teacher forcing
                gt = target[:, t]
                valid = gt >= 0
                gtc = gt.clamp(min=0)
                upd = torch.zeros_like(m)
                upd[idx, gtc] = valid
                m = m | upd
                chosen = h[idx, gtc]                                      # (B, H)
            else:                                                        # greedy decoding
                choice = scores.argmax(dim=-1)
                m = m.scatter(1, choice.unsqueeze(1), True)
                chosen = h[idx, choice]
            dec_input = chosen

            # Coupled ETA for the chosen stop.
            eta = self.eta_regressor(torch.cat([dec_h, chosen], dim=-1)).squeeze(-1)  # (B,)
            etas.append(eta)

        return torch.stack(route_logits, dim=1), torch.stack(etas, dim=1)

    @torch.no_grad()
    def beam_decode(self, x, mask=None, beam=5):
        """Batched beam search over the route pointer. Returns best visit order
        per sample, shape (B, N) of input-node indices (route only)."""
        B, N, _ = x.shape
        device = x.device
        h = F.relu(self.feature_mlp(x))                      # (B, N, H)
        ref = self.route_pointer(h)                          # (B, N, H)
        base_mask = (torch.zeros(B, N, dtype=torch.bool, device=device)
                     if mask is None else mask.clone().bool())

        h_b = h.repeat_interleave(beam, dim=0)               # (Bb, N, H)
        ref_b = ref.repeat_interleave(beam, dim=0)
        m = base_mask.repeat_interleave(beam, dim=0)
        Bb = B * beam
        dec_h = h_b.mean(1)
        dec_c = torch.zeros_like(dec_h)
        dec_input = h_b.mean(1)
        seqs = torch.zeros(Bb, N, dtype=torch.long, device=device)
        score = torch.full((B, beam), float('-inf'), device=device)
        score[:, 0] = 0.0
        score = score.view(Bb)
        rows = torch.arange(Bb, device=device)

        for t in range(N):
            dec_h, dec_c = self.route_lstm(dec_input, (dec_h, dec_c))
            sc = torch.matmul(ref_b, dec_h.unsqueeze(-1)).squeeze(-1)     # (Bb, N)
            sc = sc.masked_fill(m, float('-inf'))
            logp = torch.log_softmax(torch.nan_to_num(sc, neginf=-1e9), dim=-1)
            cand = (score.unsqueeze(1) + logp).view(B, beam * N)
            topv, topi = cand.topk(beam, dim=-1)
            beam_id = topi // N
            node = (topi % N).reshape(Bb)
            parent = (torch.arange(B, device=device).unsqueeze(1) * beam + beam_id).reshape(Bb)
            dec_h, dec_c = dec_h[parent], dec_c[parent]
            m = m[parent].clone()
            seqs = seqs[parent].clone()
            score = topv.reshape(Bb)
            seqs[rows, t] = node
            m[rows, node] = True
            dec_input = h_b[rows, node]

        return seqs.view(B, beam, N)[:, 0, :]
