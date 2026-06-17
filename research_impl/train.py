# -*- coding: utf-8 -*-
"""
Training harness for the map-aware delivery models.

Trains any model in research_impl/algorithms on the tensors built by
research_impl/dataset/dataset.py, saves weights, and reports the route/ETA
metrics from pre_processing/utils.Metric on the val/test split.

The dedicated evaluation runner lives in research_impl/evaluation/eval.py; this
file owns training + a quick metric read-out so the loop is observable.

Usage:
    python -m research_impl.train --model graph2route --city chongqing --epochs 20
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from research_impl.pre_processing.utils import ws, dir_check, to_device, set_seed, Metric
from research_impl.algorithms import (
    STGCN, TGCN, MapAwareGraph2Route, MapAwareM2G4RTP, MapAwareDRL4Route, MapAwareFDNet,
)

ROUTE_LOGIT_MODELS = {'graph2route', 'fdnet'}
SCORE_MODELS = {'m2g4rtp'}
POLICY_MODELS = {'drl4route'}
ETA_MODELS = {'stgcn', 'tgcn'}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class NpyDataset(Dataset):
    def __init__(self, path):
        self.d = np.load(path, allow_pickle=True).item()
        self.n = len(self.d['length'])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: self.d[k][i] for k in self.d}


def collate(batch):
    out = {}
    for k in batch[0]:
        arr = np.stack([b[k] for b in batch])
        if k == 'mask':
            out[k] = torch.as_tensor(arr, dtype=torch.bool)
        elif k in ('label', 'length', 'courier_id'):
            out[k] = torch.as_tensor(arr, dtype=torch.long)
        else:
            out[k] = torch.as_tensor(arr, dtype=torch.float32)
    return out


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name, cfg):
    Fn, H, Nmax, C = cfg['F'], cfg['hidden'], cfg['Nmax'], cfg['n_couriers']
    return {
        'stgcn':       lambda: STGCN(Nmax, Fn, H, 1),
        'tgcn':        lambda: TGCN(Nmax, Fn, H, 1),
        'graph2route': lambda: MapAwareGraph2Route(Nmax, Fn, H, C),
        'm2g4rtp':     lambda: MapAwareM2G4RTP(Fn, H),
        'drl4route':   lambda: MapAwareDRL4Route(Fn, H),
        'fdnet':       lambda: MapAwareFDNet(Fn, H),
    }[name]()


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _seq_ce(logits, label):
    """Cross-entropy of step-wise pointer logits (B,N,N) vs visit order (B,N)."""
    B, N, _ = logits.shape
    logits = torch.nan_to_num(logits, neginf=-1e9)
    return F.cross_entropy(logits.reshape(B * N, N), label.reshape(B * N), ignore_index=-1)


def _eta_mse(pred, target, valid):
    diff = (pred - target) ** 2 * valid
    return diff.sum() / valid.sum().clamp(min=1)


def _score_rank_loss(scores, label, length, valid):
    """Push earlier-visited stops to higher scores (desirability regression)."""
    target = torch.zeros_like(scores)
    for b in range(scores.shape[0]):
        Ln = int(length[b])
        nodes = label[b, :Ln]
        target[b, nodes] = 1.0 - torch.arange(Ln, device=scores.device).float() / max(Ln, 1)
    return _eta_mse(scores, target, valid)


def _masked_argsort(scores, valid):
    s = scores.clone()
    s[~valid] = float('-inf')
    return s.argsort(dim=-1, descending=True)


# ---------------------------------------------------------------------------
# Forward + loss + predicted sequence (per model family)
# ---------------------------------------------------------------------------

def forward_step(name, model, batch):
    V, A, L, mask = batch['V'], batch['A'], batch['L'], batch['mask']
    label, eta_label, cid = batch['label'], batch['eta_label'], batch['courier_id']
    length, valid = batch['length'], (~mask).float()
    eta_pred = None
    step_scores = None  # per-step candidate logits (pointer models) for scores-based HR@K

    if name == 'graph2route':
        logits = model(V, L, cid, mask)
        loss = _seq_ce(logits, label)
        pred_seq = logits.argmax(-1)
        step_scores = logits
    elif name == 'fdnet':
        logits, eta_pred = model(V, mask)
        loss = _seq_ce(logits, label) + _eta_mse(eta_pred, eta_label, valid)
        pred_seq = logits.argmax(-1)
        step_scores = logits
    elif name in SCORE_MODELS:
        scores = model(V, A)
        loss = _score_rank_loss(scores, label, length, valid)
        pred_seq = _masked_argsort(scores, mask == 0)
    elif name in POLICY_MODELS:
        probs, _ = model(V, A, mask)
        first = label[:, :1].clamp(min=0)
        loss = -torch.log(probs.gather(1, first) + 1e-9).mean()   # imitation of first move
        pred_seq = _masked_argsort(probs, mask == 0)
    else:  # ETA regressors -> route by predicted finish time
        out = model(V, L if name == 'stgcn' else A).squeeze(-1)
        eta_pred = out
        loss = _eta_mse(out, eta_label, valid)
        pred_seq = _masked_argsort(-out, mask == 0)

    return loss, pred_seq, eta_pred, step_scores


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(name, model, loader, device):
    model.eval()
    metric = Metric()
    for batch in loader:
        batch = to_device(batch, device)
        _, pred_seq, eta_pred, _ = forward_step(name, model, batch)
        length = batch['length']
        for b in range(len(length)):
            Ln = int(length[b])
            metric.update_route(pred_seq[b, :Ln], batch['label'][b, :Ln])
            if eta_pred is not None:
                metric.update_eta(eta_pred[b, :Ln], batch['eta_label'][b, :Ln])
    return metric.summary()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a map-aware delivery model")
    parser.add_argument('--model', required=True,
                        choices=['stgcn', 'tgcn', 'graph2route', 'm2g4rtp', 'drl4route', 'fdnet'])
    parser.add_argument('--city', default='chongqing')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = os.path.join(ws, 'research_impl', 'dataset', args.city)

    train_ds = NpyDataset(os.path.join(data_dir, 'train.npy'))
    val_ds = NpyDataset(os.path.join(data_dir, 'val.npy'))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    cfg = {
        'F': train_ds.d['V'].shape[-1],
        'Nmax': train_ds.d['V'].shape[1],
        'hidden': args.hidden,
        'n_couriers': int(train_ds.d['courier_id'].max()) + 2,
    }
    model = build_model(args.model, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_dl:
            batch = to_device(batch, device)
            loss, _, _, _ = forward_step(args.model, model, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        val = evaluate(args.model, model, val_dl, device)
        print(f"[epoch {epoch:02d}] loss={total / max(len(train_dl),1):.4f}  val={val}")

    out = os.path.join(data_dir, 'weights', f'{args.model}.pth')
    dir_check(out)
    torch.save(model.state_dict(), out)
    with open(os.path.join(data_dir, 'weights', f'{args.model}.cfg.json'), 'w') as f:
        json.dump(cfg, f)
    print(f"[+] saved weights -> {out}")


if __name__ == "__main__":
    main()
