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

    def forward(self, x, L, courier_id, V_reach_mask=None, target=None):
        """
        x: (B, N, F) features mapped to OSM nodes
        L: Laplacian from the OSM graph (N, N) or (B, N, N)
        courier_id: (B,)
        V_reach_mask: (B, N) bool, True = stop unavailable (visited/unreachable)
        target: (B, N) ground-truth visit order; if given, decode with teacher
                forcing (training). If None, decode greedily (inference).
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

        # 4. Pointer decoding (teacher-forced when target is given)
        dec_h = enc.mean(dim=1)                              # (B, H+d_w)
        dec_c = torch.zeros_like(dec_h)
        dec_input = enc.mean(dim=1)
        idx = torch.arange(B, device=device)

        logits = []
        for t in range(N):
            dec_h, dec_c = self.decoder_rnn(dec_input, (dec_h, dec_c))
            q = self.pointer_query(dec_h).unsqueeze(1)       # (B, 1, H)
            scores = self.pointer_v(torch.tanh(q + ref)).squeeze(-1)  # (B, N)
            scores = scores.masked_fill(mask, float('-inf'))
            logits.append(scores)

            if target is not None:                           # teacher forcing
                gt = target[:, t]
                valid = gt >= 0                              # ignore padded steps
                gtc = gt.clamp(min=0)
                upd = torch.zeros_like(mask)
                upd[idx, gtc] = valid
                mask = mask | upd                            # mark GT stop visited
                dec_input = enc[idx, gtc]
            else:                                            # greedy decoding
                choice = scores.argmax(dim=-1)
                mask = mask.scatter(1, choice.unsqueeze(1), True)
                dec_input = enc[idx, choice]

        return torch.stack(logits, dim=1)                    # (B, N, N)

    @torch.no_grad()
    def beam_decode(self, x, L, courier_id, V_reach_mask=None, beam=5):
        """Batched beam search over the pointer decoder. Returns the best
        predicted visit order per sample, shape (B, N) of input-node indices."""
        B, N, _ = x.shape
        device = x.device
        h_spatial = self.spatial_encoder(x, L)
        h_temporal, _ = self.temporal_gru(h_spatial)
        w_emb = self.worker_emb(courier_id).unsqueeze(1).expand(-1, N, -1)
        enc = torch.cat([h_temporal, w_emb], dim=-1)         # (B, N, E)
        ref = self.pointer_ref(h_temporal)                   # (B, N, H)
        base_mask = (torch.zeros(B, N, dtype=torch.bool, device=device)
                     if V_reach_mask is None else V_reach_mask.clone().bool())

        # expand each sample into `beam` hypotheses
        enc_b = enc.repeat_interleave(beam, dim=0)           # (Bb, N, E)
        ref_b = ref.repeat_interleave(beam, dim=0)           # (Bb, N, H)
        mask = base_mask.repeat_interleave(beam, dim=0)      # (Bb, N)
        Bb = B * beam
        dec_h = enc_b.mean(1)
        dec_c = torch.zeros_like(dec_h)
        dec_input = enc_b.mean(1)
        seqs = torch.zeros(Bb, N, dtype=torch.long, device=device)
        score = torch.full((B, beam), float('-inf'), device=device)
        score[:, 0] = 0.0
        score = score.view(Bb)
        rows = torch.arange(Bb, device=device)

        for t in range(N):
            dec_h, dec_c = self.decoder_rnn(dec_input, (dec_h, dec_c))
            q = self.pointer_query(dec_h).unsqueeze(1)
            sc = self.pointer_v(torch.tanh(q + ref_b)).squeeze(-1)        # (Bb, N)
            sc = sc.masked_fill(mask, float('-inf'))
            logp = torch.log_softmax(torch.nan_to_num(sc, neginf=-1e9), dim=-1)
            cand = (score.unsqueeze(1) + logp).view(B, beam * N)          # (B, beam*N)
            topv, topi = cand.topk(beam, dim=-1)                          # (B, beam)
            beam_id = topi // N
            node = (topi % N).reshape(Bb)
            parent = (torch.arange(B, device=device).unsqueeze(1) * beam + beam_id).reshape(Bb)
            dec_h, dec_c = dec_h[parent], dec_c[parent]
            mask = mask[parent].clone()
            seqs = seqs[parent].clone()
            score = topv.reshape(Bb)
            seqs[rows, t] = node
            mask[rows, node] = True
            dec_input = enc_b[rows, node]

        return seqs.view(B, beam, N)[:, 0, :]                # best beam per sample
