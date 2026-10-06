"""Overtaking difficulty: one data-derived number per circuit.

GridStickiness = average Spearman correlation between grid order and finishing order
(finishers only) at this circuit, over races BEFORE this one. 1.0 = the grid is the
result (Monaco ~0.87); low = order shuffles a lot (Las Vegas ~0.44). A circuit with no
history yet falls back to the average of every earlier race.

Same 8-fold leave-one-season-out CV and seeds as rebuild/train.py.
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

# same physical circuit under different race names; 2026 "Spanish GP" is a new track (Madrid)
SAME_CIRCUIT = {'70th Anniversary Grand Prix': 'British Grand Prix', 'Styrian Grand Prix': 'Austrian Grand Prix',
                'Mexican Grand Prix': 'Mexico City Grand Prix', 'Brazilian Grand Prix': 'São Paulo Grand Prix',
                'Barcelona Grand Prix': 'Spanish Grand Prix'}


def circuit(event, year):
    if event == 'Spanish Grand Prix' and year >= 2026:
        return 'Madrid'
    return SAME_CIRCUIT.get(event, event)


raw = load_data()
fin = raw[~raw['IsDNF'].astype(bool) & raw['ActualPosition'].notna() & raw['ActualGridPosition'].notna()]
races = []
for (y, r), g in fin.groupby(['Year', 'RoundNumber']):
    races.append(dict(Year=y, RoundNumber=r, Circuit=circuit(g['EventName'].iloc[0], y),
                      rho=g['ActualGridPosition'].corr(g['ActualPosition'], method='spearman')))
races = pd.DataFrame(races).sort_values(['Year', 'RoundNumber']).reset_index(drop=True)
races['_circuit'] = races.groupby('Circuit')['rho'].transform(lambda s: s.shift(1).expanding().mean())
races['_overall'] = races['rho'].shift(1).expanding().mean()
races['GridStickiness'] = races['_circuit'].fillna(races['_overall'])
print(f'GridStickiness: {races["_circuit"].notna().mean() * 100:.0f}% of races from the circuit\'s own history, '
      f'{(races["_circuit"].isna() & races["_overall"].notna()).mean() * 100:.0f}% overall-average fallback')
print('2026 values:', races[races.Year == 2026][['Circuit', 'GridStickiness']].round(2)
      .assign(Circuit=lambda t: t.Circuit.str.replace(' Grand Prix', '')).to_string(index=False, header=False).replace('\n', ' | '))

df = split_years(filter_races(load_data()), config.TRAIN_YEARS).merge(
    races[['Year', 'RoundNumber', 'GridStickiness']], on=['Year', 'RoundNumber'], how='left')

grid = []
for vy in config.TRAIN_YEARS:
    _, va = split_train_val(df, vy)
    rk = va.groupby(['Year', 'RoundNumber'])['ActualGridPosition'].rank(method='first')
    top = va['ActualPosition'] <= 10
    grid.append((rk[top] - va.loc[top, 'ActualPosition']).abs().mean())
grid = np.array(grid)

base = list(config.FEATURE_COLS)
configs = {'reference': base, '+ GridStickiness': base + ['GridStickiness']}
results = {}
for name, cols in configs.items():
    config.FEATURE_COLS[:] = cols
    with contextlib.redirect_stdout(io.StringIO()):
        curves, importance = train.cross_validate(df)
    best = int(curves.mean(axis=0).argmin())
    results[name] = (curves[:, best], dict(zip(cols, importance)))
    print(f'done: {name}', flush=True)

ref = results['reference'][0]
print(f'\n{"":<20}{"CV mean":>8}{"vs ref":>8}{"better":>8}{"vs quali":>10}{"beats quali":>13}')
print(f'{"qualifying order":<20}{grid.mean():>8.3f}')
for name, (v, imp) in results.items():
    print(f'{name:<20}{v.mean():>8.3f}{v.mean() - ref.mean():>+8.3f}{int((v < ref).sum()):>6}/8'
          f'{v.mean() - grid.mean():>+10.3f}{int((v < grid).sum()):>11}/8')
print('\nimportance with GridStickiness:', {k: round(float(x), 3) for k, x in sorted(results['+ GridStickiness'][1].items(), key=lambda kv: -kv[1])})
print('\nper fold:  ' + '  '.join(str(y) for y in config.TRAIN_YEARS))
print(f'{"quali order":<20}' + ' '.join(f'{x:.3f}' for x in grid))
for name, (v, _) in results.items():
    print(f'{name:<20}' + ' '.join(f'{x:.3f}' for x in v))
