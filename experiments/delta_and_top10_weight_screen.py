"""Predict the CHANGE from the grid instead of the finishing position, and/or weight
top-10 finishers more in training. Scored three ways, every config vs qualifying order:

  full field       every finisher graded (nothing chosen using the result)
  true top 10      drivers who actually finished top 10 (what rebuild/train.py selects on)
  predicted top 10 the drivers each method PUT in its top 10 (what the site shows)

Delta: target = finish - grid_rank (among finishers), prediction score = grid_rank + delta.
Top-10 weight: every driver still trains, but a mistake on an actual top-10 finisher
counts w times as much in the loss (the old "top-10 only" approach threw the rest away).

The training loop copies rebuild/train.py (AdamW, ExponentialLR 0.95, train-loss
stopping, epoch picked from the CV curve on the true-top-10 score), so the
'absolute' config must reproduce rebuild's CV number.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'rebuild'))
import config
from config import TRAIN_YEARS, HIDDEN, BATCH_SIZE, LR, MAX_EPOCHS, CV_SEEDS, PATIENCE, MIN_DELTA
from data import load_data, filter_races, split_years, split_train_val, fit_preprocessing, make_X
from model import build_model

RACE = ['Year', 'RoundNumber']
df_all = split_years(filter_races(load_data()), TRAIN_YEARS)
df_all['grid_rank'] = df_all.groupby(RACE)['ActualGridPosition'].rank(method='first')


def scores(df, s):
    """all three scores for prediction scores s (lower = further forward)."""
    pr = pd.Series(s, index=df.index).groupby([df['Year'], df['RoundNumber']]).rank(method='first')
    e = (pr - df['ActualPosition']).abs()
    g = (df['grid_rank'] - df['ActualPosition']).abs()
    t10 = df['ActualPosition'] <= 10
    return dict(full=e.mean(), true10=e[t10].mean(), pred10=e[pr <= 10].mean(), pred10_exact=(e[pr <= 10] == 0).mean(),
                g_full=g.mean(), g_true10=g[t10].mean(), g_pred10=g[df['grid_rank'] <= 10].mean(),
                g_pred10_exact=(g[df['grid_rank'] <= 10] == 0).mean())


def run(df_tr, df_va, X_tr, X_va, delta, w10):
    y = df_tr['ActualPosition'].values - (df_tr['grid_rank'].values if delta else 0)
    w = np.where(df_tr['ActualPosition'].values <= 10, w10, 1.0)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y), torch.FloatTensor(w)),
                        batch_size=BATCH_SIZE, shuffle=True)
    model = build_model(X_tr.shape[1], HIDDEN)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
    best, stale, hist = float('inf'), 0, []
    Xv = torch.FloatTensor(X_va)
    for epoch in range(MAX_EPOCHS):
        model.train(); tot = 0
        for xb, yb, wb in loader:
            opt.zero_grad()
            loss = (wb * (model(xb).squeeze() - yb) ** 2).mean()
            loss.backward(); opt.step(); tot += loss.item()
        tl = tot / len(loader)
        if tl < best - MIN_DELTA:
            best, stale = tl, 0
        else:
            stale += 1
        model.eval()
        with torch.no_grad():
            p = model(Xv).squeeze().numpy()
        hist.append(scores(df_va, p + (df_va['grid_rank'].values if delta else 0)))
        sch.step()
        if stale >= PATIENCE:
            break
    hist += [hist[-1]] * (MAX_EPOCHS - len(hist))
    return hist


configs = {
    'absolute (current)': (False, 1.0),
    'absolute + top-10 x2': (False, 2.0),
    'delta': (True, 1.0),
    'delta + top-10 x2': (True, 2.0),
    'delta + top-10 x3': (True, 3.0),
}
only = sys.argv[1:]
results = {}
for name, (delta, w10) in configs.items():
    if only and not any(name.startswith(o) for o in only):
        continue
    per_fold = []                                           # [fold][epoch] -> dict, averaged over seeds
    for vy in TRAIN_YEARS:
        tr, va = split_train_val(df_all, vy)
        med, sc = fit_preprocessing(tr)
        X_tr, X_va = make_X(tr, med, sc), make_X(va, med, sc)
        seed_hists = []
        for seed in CV_SEEDS:
            torch.manual_seed(1000 * seed + vy)
            seed_hists.append(run(tr, va, X_tr, X_va, delta, w10))
        per_fold.append([{k: np.mean([h[e][k] for h in seed_hists]) for k in seed_hists[0][0]} for e in range(MAX_EPOCHS)])
    curve = np.array([[f[e]['true10'] for e in range(MAX_EPOCHS)] for f in per_fold]).mean(0)
    best = int(curve.argmin())
    results[name] = {k: np.array([f[best][k] for f in per_fold]) for k in per_fold[0][0]}
    print(f'done: {name}  (epoch {best + 1})', flush=True)

first = next(iter(results.values()))
print(f'\n8-season CV (lower is better). qualifying order: full {first["g_full"].mean():.3f} | '
      f'true top 10 {first["g_true10"].mean():.3f} | predicted top 10 {first["g_pred10"].mean():.3f} '
      f'(exact {first["g_pred10_exact"].mean() * 100:.1f}%)\n')
print(f'{"":<24}{"full field":>16}{"true top 10":>16}{"predicted top 10":>22}{"exact (pred 10)":>17}')
for name, r in results.items():
    cell = lambda k, g: f'{r[k].mean():.3f} ({r[k].mean() - r[g].mean():+.3f})'
    print(f'{name:<24}{cell("full", "g_full"):>16}{cell("true10", "g_true10"):>16}{cell("pred10", "g_pred10"):>22}'
          f'{r["pred10_exact"].mean() * 100:>14.1f}%')
print('\n(+/-) = vs qualifying order; negative = better than the grid')
print('\nseasons beating qualifying order on the PREDICTED-top-10 score:')
for name, r in results.items():
    print(f'  {name:<24}{int((r["pred10"] < r["g_pred10"]).sum())}/8   per season: '
          + ' '.join(f'{a - b:+.2f}' for a, b in zip(r['pred10'], r['g_pred10'])))
