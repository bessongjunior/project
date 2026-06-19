# -*- coding: utf-8 -*-
"""
Evaluation runner for the map-aware delivery models.

Loads the held-out test split, runs each trained model AND the routing baselines,
reports the evaluation.md metrics via pre_processing/utils.Metric, and assigns the
A-D grade (evaluation.md §3) relative to the baselines.

  route : HR@1/3/5, KRC, LSD (metres), ED
  ETA   : MAE, RMSE, MAPE, ACC@10/20   (models only; baselines are route-only)
  speed : inference time per 1k samples

Grade (evaluation.md §3), per model, relative to baselines:
  A : beats OR-Tools in speed AND MAPE < 10%
  B : beats LightGBM (route KRC) AND HR@3 > 75%
  D : worse than Distance-Greedy (route KRC)
  C : otherwise

Baseline routing setup: the courier's current position (ground-truth first stop)
is known; each baseline orders the remaining stops from there.

Usage:
    python -m research_impl.evaluation.eval --city chongqing
    python -m research_impl.evaluation.eval --city chongqing --models graph2route fdnet
"""
import os
import csv
import json
import time
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from research_impl.pre_processing.utils import ws, to_device, Metric
from research_impl.train import NpyDataset, collate, build_model, forward_step
from research_impl.algorithms.baseline import (
    run_baseline, LightGBMBaseline, lightgbm_available,
)

ALL_MODELS = ['stgcn', 'tgcn', 'graph2route', 'm2g4rtp', 'drl4route', 'fdnet']
HEURISTIC_BASELINES = ['distance_greedy', 'time_greedy', 'ortools']
# accept-time is cyclically encoded at V columns 2 (sin) and 3 (cos);
# recover a monotonic time-of-day signal for the time-greedy baseline via atan2.
ACCEPT_SIN_COL, ACCEPT_COS_COL = 2, 3

# evaluation.md absolute thresholds (baseline-free gates): (metric, thr, higher_is_better)
GATES = [('hr@3', 0.70, True), ('krc', 0.60, True), ('ed', 2.0, False), ('mape', 15.0, False)]


def _data_dir(city):
    return os.path.join(ws, 'research_impl', 'dataset', city)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(name, city, device, batch=64, rush=False, beam=1):
    d = _data_dir(city)
    test_path, wpath = os.path.join(d, 'test.npy'), os.path.join(d, 'weights', f'{name}.pth')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"missing {test_path} (run tensorize first)")
    if not os.path.exists(wpath):
        return None

    ds = NpyDataset(test_path)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate)

    cfg_path = os.path.join(d, 'weights', f'{name}.cfg.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = {'F': ds.d['V'].shape[-1], 'Nmax': ds.d['V'].shape[1],
               'hidden': 64, 'n_couriers': int(ds.d['courier_id'].max()) + 2}

    model = build_model(name, cfg).to(device)
    model.load_state_dict(torch.load(wpath, map_location=device))
    model.eval()

    metric, n, t0 = Metric(), 0, time.perf_counter()
    rush_metric, rush_n = (Metric() if rush else None), 0
    for b in dl:
        b = to_device(b, device)
        _, pred_seq, eta_pred, step_scores = forward_step(name, model, b)
        if beam > 1:                                # beam search for all pointer models
            if name == 'graph2route':
                pred_seq = model.beam_decode(b['V'], b['L'], b['courier_id'], b['mask'], beam=beam)
            elif name == 'fdnet':
                pred_seq = model.beam_decode(b['V'], b['mask'], beam=beam)
            elif name == 'stgcn':
                pred_seq = model.beam_decode(b['V'], b['L'], b['mask'], beam=beam)
            else:                                   # tgcn, m2g4rtp, drl4route (GCN over A)
                pred_seq = model.beam_decode(b['V'], b['A'], b['mask'], beam=beam)
            step_scores = None                      # sequence output -> positional HR@K
        coords, is_rush = b.get('coords'), b.get('is_rush')
        for i in range(len(b['length'])):
            Ln = int(b['length'][i])
            kw = {}
            if coords is not None:
                kw['coords'] = coords[i, :Ln]
            if step_scores is not None:                  # scores-based HR@K (pointer models)
                kw['scores'] = step_scores[i, :Ln]
            metric.update_route(pred_seq[i, :Ln], b['label'][i, :Ln], **kw)
            if eta_pred is not None:
                metric.update_eta(eta_pred[i, :Ln], b['eta_label'][i, :Ln])
                if rush_metric is not None and is_rush is not None and bool(is_rush[i]):
                    rush_metric.update_eta(eta_pred[i, :Ln], b['eta_label'][i, :Ln])
                    rush_n += 1
            n += 1
    res = metric.summary()
    res['sec_per_1k'] = round(1000.0 * (time.perf_counter() - t0) / max(n, 1), 4)
    if rush_metric is not None:
        res['rush_mape'] = rush_metric.summary().get('mape')
        res['rush_n'] = rush_n
    return res


# ---------------------------------------------------------------------------
# Baselines (numpy; route only)
# ---------------------------------------------------------------------------

def evaluate_baseline(name, city):
    d = _data_dir(city)
    test = np.load(os.path.join(d, 'test.npy'), allow_pickle=True).item()

    lgbm = None
    if name == 'lightgbm':
        if not lightgbm_available():
            return None
        train = np.load(os.path.join(d, 'train.npy'), allow_pickle=True).item()
        lgbm = LightGBMBaseline().fit(train)

    V, coords, label, length = test['V'], test['coords'], test['label'], test['length']
    metric, n, t0 = Metric(), 0, time.perf_counter()
    for i in range(len(length)):
        Ln = int(length[i])
        c = coords[i, :Ln]
        start = int(label[i, 0])
        if name == 'lightgbm':
            pred = lgbm.predict_seq(V[i], Ln)
        elif name == 'time_greedy':
            # mod 2*pi so afternoon angles don't wrap negative (preserves time order)
            t_sig = np.mod(np.arctan2(V[i, :Ln, ACCEPT_SIN_COL], V[i, :Ln, ACCEPT_COS_COL]),
                           2 * np.pi)
            pred = run_baseline('time_greedy', c, times=t_sig, start=start)
        else:
            pred = run_baseline(name, c, start=start)
        metric.update_route(pred, label[i, :Ln], coords=c)
        n += 1
    res = metric.summary()
    res['sec_per_1k'] = round(1000.0 * (time.perf_counter() - t0) / max(n, 1), 4)
    return res


# ---------------------------------------------------------------------------
# Grading (evaluation.md §3)
# ---------------------------------------------------------------------------

def assign_grade(m, bl):
    krc, hr3 = m.get('krc', -1.0), m.get('hr@3', 0.0)
    mape, speed = m.get('mape', 1e9), m.get('sec_per_1k', 1e9)
    dg, ort, lgbm = bl.get('distance_greedy'), bl.get('ortools'), bl.get('lightgbm')

    if ort and speed < ort['sec_per_1k'] and mape < 10.0:
        return 'A'
    if lgbm and krc > lgbm['krc'] and hr3 > 0.75:
        return 'B'
    if dg and krc < dg['krc']:
        return 'D'
    return 'C'


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

COLS = ['hr@1', 'hr@3', 'hr@5', 'krc', 'lsd', 'ed', 'mae', 'rmse', 'mape',
        'acc@10', 'acc@20', 'sec_per_1k']


def _table_str(title, rows):
    lines = [f"=== {title} ===",
             "name".ljust(16) + "".join(c.rjust(10) for c in COLS)]
    lines.append("-" * len(lines[1]))
    for name, r in rows.items():
        line = name.ljust(16)
        for c in COLS:
            v = r.get(c)
            line += ("-" if v is None else f"{v:.3f}").rjust(10)
        lines.append(line)
    return "\n".join(lines)


def compute_grades(models, baselines):
    """Per-model grade + absolute-gate pass/fail."""
    out = {}
    for m, r in models.items():
        gates = {}
        for key, thr, higher in GATES:
            if key in r:
                gates[key] = bool(r[key] >= thr if higher else r[key] <= thr)
        out[m] = {'grade': assign_grade(r, baselines) if baselines else None,
                  'gates': gates}
    return out


def build_report(city, models, baselines, grades, rush_on):
    """The exact text printed to console (and saved to summary.txt)."""
    blocks = []
    if baselines:
        blocks.append(_table_str(f"{city} | baselines (test)", baselines))
    if models:
        blocks.append(_table_str(f"{city} | models (test)", models))

    grade_lines = ["=== grade (evaluation.md §3) + absolute gates ==="]
    for m in models:
        g = grades[m]['grade'] or "n/a (no baselines)"
        gate_str = " ".join(f"{k}{'OK' if ok else 'X'}" for k, ok in grades[m]['gates'].items())
        grade_lines.append(f"{m.ljust(16)} grade={g}   {gate_str}")
    blocks.append("\n".join(grade_lines))

    if rush_on:
        rl = ["=== rush-hour Temporal gate (Integration.md: MAPE < 12%) ==="]
        for m, r in models.items():
            rm = r.get('rush_mape')
            if rm is None:
                rl.append(f"{m.ljust(16)} (no ETA output or no is_rush field — rebuild tensors)")
            else:
                rl.append(f"{m.ljust(16)} rush_MAPE={rm:.2f}  n={r.get('rush_n', 0)}  "
                          f"{'OK' if rm < 12.0 else 'X'}")
        blocks.append("\n".join(rl))

    return "\n\n".join(blocks)


def _write_metrics_csv(path, models, baselines):
    fields = ['kind', 'name'] + COLS + ['rush_mape']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for kind, rows in (('baseline', baselines), ('model', models)):
            for name, r in rows.items():
                row = {'kind': kind, 'name': name, 'rush_mape': r.get('rush_mape')}
                row.update({c: r.get(c) for c in COLS})
                w.writerow(row)


def _write_grades_json(path, city, ts, models, baselines, grades):
    payload = {
        'city': city,
        'timestamp': ts,
        'models': {m: {'grade': grades[m]['grade'], 'gates': grades[m]['gates'],
                       'metrics': models[m]} for m in models},
        'baselines': dict(baselines),
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def write_outputs(city, models, baselines, grades, report):
    """Write metrics.csv / grades.json / summary.txt to a timestamped dir + latest/."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.join(ws, 'research_impl', 'evaluation', 'outputs', city)
    targets = [os.path.join(base, ts), os.path.join(base, 'latest')]
    for d in targets:
        os.makedirs(d, exist_ok=True)
        _write_metrics_csv(os.path.join(d, 'metrics.csv'), models, baselines)
        _write_grades_json(os.path.join(d, 'grades.json'), city, ts, models, baselines, grades)
        with open(os.path.join(d, 'summary.txt'), 'w') as f:
            f.write(f"# {city}  |  {ts}\n\n{report}\n")
    return targets[0]


def main():
    p = argparse.ArgumentParser(description="Evaluate + grade delivery models")
    p.add_argument('--city', default='chongqing')
    p.add_argument('--models', nargs='+', default=ALL_MODELS)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--no-baselines', action='store_true')
    p.add_argument('--rush-hour', dest='rush_hour', action='store_true',
                   help="also report peak-hour MAPE (Integration.md Temporal gate)")
    p.add_argument('--beam', type=int, default=1,
                   help="beam width for pointer models (graph2route, fdnet); 1 = greedy")
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {}
    for m in args.models:
        try:
            r = evaluate_model(m, args.city, device, args.batch, rush=args.rush_hour, beam=args.beam)
        except FileNotFoundError as e:
            print(f"[!] {m}: {e}"); continue
        if r is None:
            print(f"[-] {m}: no trained weights, skipped"); continue
        models[m] = r

    baselines = {}
    if not args.no_baselines:
        for b in HEURISTIC_BASELINES + ['lightgbm']:
            try:
                r = evaluate_baseline(b, args.city)
            except FileNotFoundError as e:
                print(f"[!] baseline {b}: {e}"); continue
            if r is None:
                print(f"[-] baseline {b}: unavailable, skipped"); continue
            baselines[b] = r

    if not models and not baselines:
        print("Nothing to evaluate. Train a model and/or build the dataset first.")
        return

    grades = compute_grades(models, baselines)
    report = build_report(args.city, models, baselines, grades, args.rush_hour)
    print("\n" + report)

    out_dir = write_outputs(args.city, models, baselines, grades, report)
    print(f"\n[+] outputs written -> {out_dir}  (and latest/)")


if __name__ == "__main__":
    main()
