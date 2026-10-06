"""Wins features again, now that the rebuild uses the REAL grid (ActualGridPosition).

With the season-average grid, no wins feature helped. The real grid is a much stronger
input, so the question changes: does knowing who usually wins add anything once the
model already knows the starting order? Same 8-fold CV and seeds as rebuild/train.py.
Also reports the qualifying-order baseline (rank by real grid) on the same folds.
"""
import contextlib, io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'rebuild'))
import config
from data import load_data, filter_races, split_years, split_train_val
import train

raw = load_data().sort_values(['Year', 'RoundNumber']).reset_index(drop=True)
raw['_win'] = (raw['ActualPosition'] == 1).astype(int)
raw['WinsThisSeason'] = raw.groupby(['DriverName', 'Year'])['_win'].transform(lambda s: s.shift(1).fillna(0).cumsum())
per_year = raw.groupby(['DriverName', 'Year'])['_win'].sum()
pre = json.loads((ROOT / 'data' / 'pre2018_wins.json').read_text(encoding='utf-8'))
raw['WinsLastSeason'] = [pre.get(d, {}).get('by_year', {}).get('2017', 0) if y == 2018 else per_year.get((d, y - 1), 0)
                         for d, y in zip(raw['DriverName'], raw['Year'])]

keys = ['Year', 'RoundNumber', 'DriverName']
df = split_years(filter_races(load_data()), config.TRAIN_YEARS).merge(
    raw[keys + ['WinsThisSeason', 'WinsLastSeason']], on=keys, how='left')

# qualifying-order baseline, scored exactly like ranked_top10_mae
grid = []
for vy in config.TRAIN_YEARS:
    _, va = split_train_val(df, vy)
    r = va.groupby(['Year', 'RoundNumber'])['ActualGridPosition'].rank(method='first')
    top = va['ActualPosition'] <= 10
    grid.append((r[top] - va.loc[top, 'ActualPosition']).abs().mean())
grid = np.array(grid)

base = list(config.FEATURE_COLS)
configs = {
    'reference (current 6 features)': base,
    '+ CareerWins': base + ['CareerWins'],
    '+ WinsLast3Years': base + ['WinsLast3Years'],
    '+ WinsThisSeason': base + ['WinsThisSeason'],
    '+ WinsThisSeason + WinsLastSeason': base + ['WinsThisSeason', 'WinsLastSeason'],
}

results = {}
for name, cols in configs.items():
    config.FEATURE_COLS[:] = cols
    with contextlib.redirect_stdout(io.StringIO()):
        curves, importance = train.cross_validate(df)
    best = int(curves.mean(axis=0).argmin())
    results[name] = (curves[:, best], dict(zip(cols, importance)))
    print(f'done: {name}', flush=True)

ref = results['reference (current 6 features)'][0]
print(f'\n{"":<38}{"CV mean":>8}{"vs ref":>8}{"better":>8}{"vs quali":>10}{"beats quali":>13}   wins importance')
print(f'{"qualifying order (real grid)":<38}{grid.mean():>8.3f}{"":>8}{"":>8}{0:>+10.3f}{"-":>13}')
for name, (v, imp) in results.items():
    w = {k: round(float(x), 3) for k, x in imp.items() if 'Win' in k}
    print(f'{name:<38}{v.mean():>8.3f}{v.mean() - ref.mean():>+8.3f}{int((v < ref).sum()):>6}/8'
          f'{v.mean() - grid.mean():>+10.3f}{int((v < grid).sum()):>11}/8   {w}')
print('\nimportance, reference:', {k: round(float(x), 3) for k, x in sorted(results['reference (current 6 features)'][1].items(), key=lambda kv: -kv[1])})
print('\nper fold:  ' + '  '.join(str(y) for y in config.TRAIN_YEARS))
print(f'{"quali order":<30}' + ' '.join(f'{x:.3f}' for x in grid))
for name, (v, _) in results.items():
    print(f'{name[:30]:<30}' + ' '.join(f'{x:.3f}' for x in v))
