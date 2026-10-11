"""Audit fixes 1, 4, 5 measured together, for both rebuild models.

  1. ensemble   - average the predictions of 5 seeds instead of using one model
  4. points     - PointsShare from the running total of Points (race + sprint) instead
                  of SeasonPoints (which leaves sprints out, keeps the old fastest-lap point)
  5. epochs     - pick the epoch count on the FULL-FIELD score instead of the true-top-10
                  score (which chooses who to grade using the result)

Training loop copies rebuild/train.py (AdamW, ExponentialLR 0.95, train-loss stopping,
top-10 weight). 8-fold leave-one-season-out. For every config, every epoch, every seed we
keep the validation predictions, so single-model and ensemble scores come from the same runs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'rebuild'))
from config import (MODELS, TRAIN_YEARS, HIDDEN, BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE, MIN_DELTA,
                    TOP10_WEIGHT)
from data import load_data, filter_races, split_years, split_train_val, fit_preprocessing, make_X
from model import build_model

SEEDS = [0, 1, 2, 3, 4]
RACE = ['Year', 'RoundNumber']

raw = load_data()
pts = raw['Points'].fillna(0)
raw['SeasonPointsAll'] = pts.groupby([raw['DriverName'], raw['Year']]).transform(lambda s: s.cumsum().shift(1).fillna(0))
lead = raw.groupby(RACE)['SeasonPointsAll'].transform('max')
raw['PointsShareAll'] = (raw['SeasonPointsAll'] / lead.replace(0, np.nan)).fillna(0)
keys = RACE + ['DriverName']
df = split_years(filter_races(raw), TRAIN_YEARS).reset_index(drop=True)
df['grid_rank'] = df.groupby(RACE)['ActualGridPosition'].rank(method='first')


def scores(va, s):
    pr = pd.Series(s, index=va.index).groupby([va['Year'], va['RoundNumber']]).rank(method='first')
    e = (pr - va['ActualPosition']).abs()
    return e.mean(), e[va['ActualPosition'] <= 10].mean(), e[pr <= 10].mean()      # full, true10, pred10


def run(X_tr, y, X_va):
    """train like rebuild/train.py; return validation predictions after every epoch."""
    w = np.where(y <= 10, TOP10_WEIGHT, 1.0)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y), torch.FloatTensor(w)),
                        batch_size=BATCH_SIZE, shuffle=True)
    model = build_model(X_tr.shape[1], HIDDEN)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
    best, stale, preds = float('inf'), 0, []
    Xv = torch.FloatTensor(X_va)
    for _ in range(MAX_EPOCHS):
        model.train(); tot = 0
        for xb, yb, wb in loader:
            opt.zero_grad()
            loss = (wb * (model(xb).squeeze() - yb) ** 2).mean()
            loss.backward(); opt.step(); tot += loss.item()
        tl = tot / len(loader)
        best, stale = (tl, 0) if tl < best - MIN_DELTA else (best, stale + 1)
        model.eval()
        with torch.no_grad():
            preds.append(model(Xv).squeeze(-1).numpy())
        sch.step()
        if stale >= PATIENCE:
            break
    preds += [preds[-1]] * (MAX_EPOCHS - len(preds))
    return np.array(preds)                                     # [epoch, row]


def evaluate_config(features):
    # per fold: preds[seed][epoch][row]
    folds = []
    for vy in TRAIN_YEARS:
        tr, va = split_train_val(df, vy)
        med, sc = fit_preprocessing(tr, features)
        X_tr, X_va = make_X(tr, med, sc, features), make_X(va, med, sc, features)
        P = []
        for seed in SEEDS:
            torch.manual_seed(1000 * seed + vy)
            P.append(run(X_tr, tr['ActualPosition'].values, X_va))
        folds.append((va, np.array(P)))
    # single-model score per (fold, seed, epoch) and ensemble score per (fold, epoch)
    single = np.array([[[scores(va, P[s, e]) for e in range(MAX_EPOCHS)] for s in range(len(SEEDS))] for va, P in folds])
    ens = np.array([[scores(va, P[:, e].mean(0)) for e in range(MAX_EPOCHS)] for va, P in folds])
    return single, ens       # single: [fold, seed, epoch, metric]  ens: [fold, epoch, metric]


M = ['full field', 'true top 10', 'predicted top 10']
results = {}
for model_name in ['postquali', 'prequali']:
    base = MODELS[model_name]['features']
    for pv, feats in [('SeasonPoints (now)', base),
                      ('Points incl. sprints', ['PointsShareAll' if c == 'PointsShare' else c for c in base])]:
        single, ens = evaluate_config(feats)
        for sel_name, sel in [('true top 10 (now)', 1), ('full field', 0)]:
            ep_single = int(single[:, :, :, sel].mean(axis=(0, 1)).argmin())
            ep_ens = int(ens[:, :, sel].mean(axis=0).argmin())
            s = single[:, :, ep_single, :]                      # [fold, seed, metric]
            results[(model_name, pv, sel_name)] = dict(
                single=s.mean(axis=(0, 1)), seed_range=s.mean(axis=0).max(0) - s.mean(axis=0).min(0),
                ens=ens[:, ep_ens, :].mean(axis=0), ens_folds=ens[:, ep_ens, :], single_folds=s.mean(axis=1),
                ep=(ep_single + 1, ep_ens + 1))
        print(f'done: {model_name} / {pv}', flush=True)

for model_name in ['postquali', 'prequali']:
    print(f'\n================ {model_name} (8-season CV, lower is better) ================')
    print(f'{"points":<22}{"epochs picked on":<20}{"":<10}' + ''.join(f'{m:>18}' for m in M) + '   epochs')
    ref = results[(model_name, 'SeasonPoints (now)', 'true top 10 (now)')]['single']
    for (mn, pv, sel_name), r in results.items():
        if mn != model_name:
            continue
        print(f'{pv:<22}{sel_name:<20}{"1 model":<10}' + ''.join(f'{v:>11.3f} ({v - ref[i]:+.3f})' for i, v in enumerate(r['single']))
              + f'   {r["ep"][0]}')
        print(f'{"":<22}{"":<20}{"(seed luck)":<10}' + ''.join(f'{"+/- " + format(x / 2, ".3f"):>18}' for x in r['seed_range']))
        print(f'{"":<22}{"":<20}{"5-seed avg":<10}' + ''.join(f'{v:>11.3f} ({v - ref[i]:+.3f})' for i, v in enumerate(r['ens']))
              + f'   {r["ep"][1]}')
        wins = (r['ens_folds'] < r['single_folds']).sum(axis=0)
        print(f'{"":<52}ensemble beats single model in ' + ' / '.join(f'{int(w)}/8' for w in wins) + ' seasons')
