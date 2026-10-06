"""Rebuild vs production (the site's post-quali ensemble) on 2026, scored identically.

Both were trained on 2018-2025 only, so 2026 is a fair held-out test for both.
Same rows for everyone: 2026 finishers (rebuild's filter_races), positions renumbered
among finishers. Each model's scores are ranked within the race among those finishers.

Scores (lower is better), each vs qualifying order:
  full field        every finisher
  true top 10       drivers who actually finished top 10
  predicted top 10  the drivers each method PUT in its top 10 (what the site shows)
"""
import contextlib, io, json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
RACE = ['Year', 'RoundNumber']

# ---- rebuild ---------------------------------------------------------------
sys.path.insert(0, str(ROOT / 'rebuild'))
import config as rb_config
from data import load_data as rb_load, filter_races, split_years, make_X
from model import load_artifacts
with contextlib.redirect_stdout(io.StringIO()):
    rb_model, rb_scaler, rb_medians = load_artifacts()
rows = split_years(filter_races(rb_load()), rb_config.TEST_YEARS).reset_index(drop=True)
rb_model.eval()
with torch.no_grad():
    rows['rebuild'] = rb_model(torch.FloatTensor(make_X(rows, rb_medians, rb_scaler))).squeeze().numpy()
for m in ['config', 'data', 'model', 'train']:
    sys.modules.pop(m, None)
sys.path.remove(str(ROOT / 'rebuild'))

# ---- production post-quali ensemble (as generate_static_data uses it) --------
sys.path.insert(0, str(ROOT / 'top10'))
with contextlib.redirect_stdout(io.StringIO()):
    from train import F1NeuralNetwork, load_data as prod_load
    from feature_calculation import add_racecraft_features, add_overqual_features
    tr, te, _ = prod_load()
for d in (tr, te):
    d['SeasonAvgGrid'] = d['GridPosition']
    d['GridPosition'] = d['ActualGridPosition'].fillna(d['GridPosition'])
with contextlib.redirect_stdout(io.StringIO()):
    add_racecraft_features([tr, te])
for d in (tr, te):
    add_overqual_features(d)
meta = json.load(open(ROOT / 'models' / 'postquali_meta.json'))
feats = meta['features']
med = {c: tr[c].median() for c in feats}
scaler = pickle.load(open(ROOT / 'models' / 'scaler_top10_postquali.pkl', 'rb'))
ens = []
for f in meta['ensemble_files']:
    with contextlib.redirect_stdout(io.StringIO()):
        m = F1NeuralNetwork(input_size=len(feats), hidden_sizes=[64, 32], dropout_rate=0.4)
    m.load_state_dict(torch.load(ROOT / 'models' / f, map_location='cpu')); m.eval(); ens.append(m)
te = te[te['Year'].isin(rows['Year'].unique())].copy()
X = te[feats].copy()
for c in feats:
    X[c] = X[c].fillna(med[c])
with torch.no_grad():
    delta = np.mean([m(torch.FloatTensor(scaler.transform(X.values.astype(np.float32)))).numpy().ravel() for m in ens], axis=0)
te['production'] = te.groupby(RACE)['GridPosition'].rank(method='first').values + delta   # how the site scores a race
rows = rows.merge(te[RACE + ['DriverName', 'production']], on=RACE + ['DriverName'], how='left')
assert rows['production'].notna().all(), 'production score missing for some rows'

# ---- score both the same way -------------------------------------------------
rows['grid'] = rows['ActualGridPosition']
out = {}
for name in ['rebuild', 'production', 'grid']:
    rank = rows.groupby(RACE)[name].rank(method='first')
    err = (rank - rows['ActualPosition']).abs()
    pick = rank <= 10
    true10 = rows['ActualPosition'] <= 10
    out[name] = {
        'full field': (err.mean(), (err == 0).mean()),
        'true top 10': (err[true10].mean(), (err[true10] == 0).mean()),
        'predicted top 10': (err[pick].mean(), (err[pick] == 0).mean(), (err[pick] <= 1).mean()),
    }

print(f'2026: {rows.groupby(RACE).ngroups} races, {len(rows)} finishers\n')
for view in ['full field', 'true top 10', 'predicted top 10']:
    print(view)
    for name in ['rebuild', 'production', 'grid']:
        v = out[name][view]
        extra = f'  within-1 {v[2] * 100:.1f}%' if len(v) > 2 else ''
        diff = '' if name == 'grid' else f'   ({v[0] - out["grid"][view][0]:+.3f} vs grid)'
        label = 'qualifying order' if name == 'grid' else name
        print(f'  {label:<17}MAE {v[0]:.3f}  exact {v[1] * 100:.1f}%{extra}{diff}')
    print()

# per-race: who wins the predicted-top-10 score
per = []
for (y, r), g in rows.groupby(RACE):
    res = {}
    for name in ['rebuild', 'production', 'grid']:
        rank = g[name].rank(method='first'); err = (rank - g['ActualPosition']).abs()
        res[name] = err[rank <= 10].mean()
    per.append(res)
per = pd.DataFrame(per)
print('predicted top 10, race by race: rebuild better than production in '
      f'{int((per.rebuild < per.production).sum())}, worse in {int((per.rebuild > per.production).sum())}, '
      f'tied in {int((per.rebuild == per.production).sum())} of {len(per)} races')
