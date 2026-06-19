# -*- coding: utf-8 -*-
"""
Shared autoregressive pointer decoder + batched beam search.

Any model that produces node embeddings `enc` of shape (B, N, E) can route by
attaching this decoder: teacher-forced training via `forward(enc, mask, target)`
and beam-search inference via `beam_decode(enc, mask, beam)`. This lets every
map-aware model decode a sequence step by step (so beam search is meaningful),
not just emit a one-shot per-node score.
"""
import torch
import torch.nn as nn


class PointerDecoder(nn.Module):
    def __init__(self, enc_dim, hidden, predict_eta=False):
        super().__init__()
        self.cell = nn.LSTMCell(enc_dim, enc_dim)
        self.query = nn.Linear(enc_dim, hidden)
        self.ref = nn.Linear(enc_dim, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.predict_eta = predict_eta
        if predict_eta:                                    # per-step travel-time head
            self.eta_head = nn.Sequential(
                nn.Linear(enc_dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, enc, mask=None, target=None):
        """enc: (B, N, E). Returns (B, N, N) step x candidate logits, or
        (logits, eta (B, N)) when predict_eta. target -> teacher forcing."""
        B, N, _ = enc.shape
        device = enc.device
        m = (torch.zeros(B, N, dtype=torch.bool, device=device)
             if mask is None else mask.clone().bool())
        ref = self.ref(enc)
        dec_h = enc.mean(1)
        dec_c = torch.zeros_like(dec_h)
        dec_in = enc.mean(1)
        idx = torch.arange(B, device=device)

        logits, etas = [], []
        for t in range(N):
            dec_h, dec_c = self.cell(dec_in, (dec_h, dec_c))
            q = self.query(dec_h).unsqueeze(1)
            sc = self.v(torch.tanh(q + ref)).squeeze(-1).masked_fill(m, float('-inf'))
            logits.append(sc)
            if target is not None:                         # teacher forcing
                gt = target[:, t]
                valid = gt >= 0
                choice = gt.clamp(min=0)
                upd = torch.zeros_like(m)
                upd[idx, choice] = valid
                m = m | upd
            else:                                          # greedy
                choice = sc.argmax(dim=-1)
                m = m.scatter(1, choice.unsqueeze(1), True)
            chosen = enc[idx, choice]
            dec_in = chosen
            if self.predict_eta:
                etas.append(self.eta_head(torch.cat([dec_h, chosen], dim=-1)).squeeze(-1))

        seq_logits = torch.stack(logits, dim=1)
        if self.predict_eta:
            return seq_logits, torch.stack(etas, dim=1)
        return seq_logits

    @torch.no_grad()
    def beam_decode(self, enc, mask=None, beam=5):
        """enc: (B, N, E). Returns (B, N) best visit order (input-node indices)."""
        B, N, _ = enc.shape
        device = enc.device
        base = (torch.zeros(B, N, dtype=torch.bool, device=device)
                if mask is None else mask.clone().bool())
        enc_b = enc.repeat_interleave(beam, dim=0)
        ref_b = self.ref(enc_b)
        m = base.repeat_interleave(beam, dim=0)
        Bb = B * beam
        dec_h = enc_b.mean(1)
        dec_c = torch.zeros_like(dec_h)
        dec_in = enc_b.mean(1)
        seqs = torch.zeros(Bb, N, dtype=torch.long, device=device)
        score = torch.full((B, beam), float('-inf'), device=device)
        score[:, 0] = 0.0
        score = score.view(Bb)
        rows = torch.arange(Bb, device=device)

        for t in range(N):
            dec_h, dec_c = self.cell(dec_in, (dec_h, dec_c))
            q = self.query(dec_h).unsqueeze(1)
            sc = self.v(torch.tanh(q + ref_b)).squeeze(-1).masked_fill(m, float('-inf'))
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
            dec_in = enc_b[rows, node]
        return seqs.view(B, beam, N)[:, 0, :]
