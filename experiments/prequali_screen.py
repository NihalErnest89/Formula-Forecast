"""Pre-qualifying model: the post-quali feature set with the real grid replaced by a
PROJECTED grid that only uses earlier races.

ProjectedGrid = the driver's average real grid slot over this season's earlier races;
at round 1, last season's average; a rookie with neither -> median fill.
(collect_data's GridPosition falls back to the race's OWN real grid at round 1 -- a leak.)

Same 8-fold CV, seeds and training as rebuild/train.py. Baselines:
  form order        rank by ProjectedGrid (fair pre-quali baseline)
  qualifying order  rank by the real grid (what the post-quali model gets to see)
"""
import contextlib, io, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'rebuild'))
import config
from data import load_data, filter_races, split_years, split_train_val
import train

RACE = ['Year', 'RoundNumber']
raw = load_data()                                     # sorted by Year, RoundNumber (add_features)
grid = raw['ActualGridPosition']
this_season = grid.groupby([raw['DriverName'], raw['Year']]).transform(lambda s: s.expanding().mean().shift(1))
per_season = grid.groupby([raw['DriverName'], raw['Year']]).mean()
last_season = pd.Series([per_season.get((d, y - 1), np.nan) for d, y in zip(raw['DriverName'], raw['Year'])], index=raw.index)
raw['ProjectedGrid'] = this_season.fillna(last_season)

r1 = raw[raw['RoundNumber'] == 1]
print(f'round 1: ProjectedGrid equals the real grid in {np.mean(np.isclose(r1.ProjectedGrid, r1.ActualGridPosition)) * 100:.0f}% '
      f'of rows; collect_data GridPosition does in {np.mean(np.isclose(r1.GridPosition, r1.ActualGridPosition)) * 100:.0f}%')
print(f'missing (rookie at round 1 -> median): {raw.ProjectedGrid.isna().mean() * 100:.1f}%')
later = raw[raw['RoundNumber'] > 1]
print(f'rounds 2+: ProjectedGrid matches collect_data GridPosition in {np.mean(np.isclose(later.ProjectedGrid, later.GridPosition, equal_nan=True)) * 100:.0f}% of rows', flush=True)

keys = RACE + ['DriverName']
df = split_years(filter_races(load_data()), config.TRAIN_YEARS).merge(raw[keys + ['ProjectedGrid']], on=keys, how='left')


def baseline(col):
    full, true10, pred10 = [], [], []
    for vy in config.TRAIN_YEARS:
        _, va = split_train_val(df, vy)
        rk = va.groupby(RACE)[col].rank(method='first')
        e = (rk - va['ActualPosition']).abs()
        true10.append(e[va['ActualPosition'] <= 10].mean()); pred10.append(e[rk <= 10].mean())
    return np.array(true10), np.array(pred10)


post = list(config.FEATURE_COLS)
pre = ['ProjectedGrid' if c == 'ActualGridPosition' else c for c in post]
res = {}
for name, cols in [('post-quali (current)', post), ('pre-quali', pre)]:
    config.FEATURE_COLS[:] = cols
    with contextlib.redirect_stdout(io.StringIO()):
        curves, imp = train.cross_validate(df)
    v = curves[:, int(curves.mean(0).argmin())]
    res[name] = (v, dict(zip(cols, imp)))
    print(f'done: {name}', flush=True)

form_t, form_p = baseline('ProjectedGrid')
quali_t, quali_p = baseline('ActualGridPosition')
pre_v = res['pre-quali'][0]
print(f'\n8-season CV, true-top-10 error (lower is better)')
print(f'  qualifying order        {quali_t.mean():.3f}')
print(f'  post-quali model        {res["post-quali (current)"][0].mean():.3f}')
print(f'  form order (projected)  {form_t.mean():.3f}')
print(f'  pre-quali model         {pre_v.mean():.3f}   ({pre_v.mean() - form_t.mean():+.3f} vs form order, '
      f'better in {int((pre_v < form_t).sum())}/8 seasons)')
print(f'\npredicted-top-10 (site view) for the baselines: form order {form_p.mean():.3f}, qualifying order {quali_p.mean():.3f}')
print('\npre-quali importance:', {k: round(float(x), 3) for k, x in sorted(res['pre-quali'][1].items(), key=lambda kv: -kv[1])})
print('\nper fold:  ' + '  '.join(str(y) for y in config.TRAIN_YEARS))
print(f'{"form order":<16}' + ' '.join(f'{x:.3f}' for x in form_t))
print(f'{"pre-quali":<16}' + ' '.join(f'{x:.3f}' for x in pre_v))
