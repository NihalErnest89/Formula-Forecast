"""SeasonPoints vs SeasonStanding vs points-per-race, in the rebuild's own CV.

SeasonPoints grows through the season (0 at round 1, ~500 at the end), so the same
number means different things at different times. Standing (1-22) and points-per-race
don't drift. Same 8-fold leave-one-season-out CV and seeds as rebuild/train.py, so the
reference row should reproduce the reference score.
"""
import contextlib, io, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'rebuild'))
import config
from data import load_data, filter_races, split_years
import train

df = split_years(filter_races(load_data()), config.TRAIN_YEARS)
df['PointsPerRace'] = df['SeasonPoints'] / (df['RoundNumber'] - 1).clip(lower=1)   # round 1 -> 0

base = list(config.FEATURE_COLS)
swap = lambda new: [new if c == 'SeasonPoints' else c for c in base]
configs = {
    'reference (SeasonPoints)': base,
    'SeasonPoints -> SeasonStanding': swap('SeasonStanding'),
    'SeasonPoints -> PointsPerRace': swap('PointsPerRace'),
    'drop SeasonPoints entirely': [c for c in base if c != 'SeasonPoints'],
}

results = {}
for name, cols in configs.items():
    config.FEATURE_COLS[:] = cols                     # same list object everywhere it was imported
    with contextlib.redirect_stdout(io.StringIO()):
        curves, importance = train.cross_validate(df)
    best = int(curves.mean(axis=0).argmin())
    results[name] = (curves[:, best], best + 1)
    print(f'{name:<34} done', flush=True)

ref = results['reference (SeasonPoints)'][0]
print(f'\n{"":<34}{"CV mean":>8}{"+/-":>7}{"vs ref":>8}{"folds better":>14}{"best ep":>9}')
for name, (v, ep) in results.items():
    print(f'{name:<34}{v.mean():>8.3f}{v.std():>7.3f}{v.mean() - ref.mean():>+8.3f}{int((v < ref).sum()):>11}/8{ep:>9}')
print('\nper fold:  ' + '  '.join(str(y) for y in config.TRAIN_YEARS))
for name, (v, _) in results.items():
    print(f'{name[:26]:<26}' + ' '.join(f'{x:.3f}' for x in v))
