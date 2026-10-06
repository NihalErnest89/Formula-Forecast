"""Which wins window? WinsLast3Years vs season-scoped versions.

WinsLast3Years carries wins across regulation changes: in 2026 it rated Verstappen
(17: 9 in 2024 + 8 in 2025 + 0 in 2026) above Antonelli (8, all in 2026). RecentForm
showed that resetting at the season start beats carrying last season over, so test the
same idea for wins. Same 8-fold leave-one-season-out CV and seeds as rebuild/train.py.

All counts use only races BEFORE the one being predicted, keyed by driver identity.
2018's "last season" (2017) comes from data/pre2018_wins.json.
"""
import contextlib, io, json, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'rebuild'))
import config
from data import load_data, filter_races, split_years
import train

raw = load_data().sort_values(['Year', 'RoundNumber']).reset_index(drop=True)
raw['_win'] = (raw['ActualPosition'] == 1).astype(int)

# wins this season BEFORE this race (resets every season)
raw['WinsThisSeason'] = raw.groupby(['DriverName', 'Year'])['_win'].transform(lambda s: s.shift(1).fillna(0).cumsum())
# races run so far this season, for a rate that doesn't grow through the year
raw['_races_before'] = raw.groupby(['DriverName', 'Year']).cumcount()
raw['WinRateThisSeason'] = raw['WinsThisSeason'] / raw['_races_before'].clip(lower=1)

# wins in the whole previous season (2017 from the pre-2018 table)
per_year = raw.groupby(['DriverName', 'Year'])['_win'].sum()
pre = json.loads((ROOT / 'data' / 'pre2018_wins.json').read_text(encoding='utf-8'))
def last_season(row):
    if row['Year'] == 2018:
        return pre.get(row['DriverName'], {}).get('by_year', {}).get('2017', 0)
    return per_year.get((row['DriverName'], row['Year'] - 1), 0)
raw['WinsLastSeason'] = raw.apply(last_season, axis=1)
raw['WinsThisPlusLast'] = raw['WinsThisSeason'] + raw['WinsLastSeason']

new = ['WinsThisSeason', 'WinRateThisSeason', 'WinsLastSeason', 'WinsThisPlusLast']
keys = ['Year', 'RoundNumber', 'DriverName']
df = split_years(filter_races(load_data()), config.TRAIN_YEARS).merge(raw[keys + new], on=keys, how='left')

base = list(config.FEATURE_COLS)
swap = lambda *cols: [c for c in base if c != 'WinsLast3Years'] + list(cols)
configs = {
    'reference (WinsLast3Years)': base,
    '-> WinsThisSeason': swap('WinsThisSeason'),
    '-> WinRateThisSeason': swap('WinRateThisSeason'),
    '-> WinsLastSeason': swap('WinsLastSeason'),
    '-> WinsThisSeason + WinsLastSeason (2 cols)': swap('WinsThisSeason', 'WinsLastSeason'),
    '-> WinsThisPlusLast (summed)': swap('WinsThisPlusLast'),
    'drop wins entirely': swap(),
}

results = {}
for name, cols in configs.items():
    config.FEATURE_COLS[:] = cols
    with contextlib.redirect_stdout(io.StringIO()):
        curves, importance = train.cross_validate(df)
    best = int(curves.mean(axis=0).argmin())
    imp = dict(zip(cols, importance))
    results[name] = (curves[:, best], imp)
    print(f'done: {name}', flush=True)

ref = results['reference (WinsLast3Years)'][0]
print(f'\n{"":<46}{"CV mean":>8}{"vs ref":>8}{"folds better":>14}   wins-feature importance')
for name, (v, imp) in results.items():
    w = {k: round(float(x), 3) for k, x in imp.items() if 'Win' in k}
    print(f'{name:<46}{v.mean():>8.3f}{v.mean() - ref.mean():>+8.3f}{int((v < ref).sum()):>11}/8   {w}')
print('\nper fold:  ' + '  '.join(str(y) for y in config.TRAIN_YEARS))
for name, (v, _) in results.items():
    print(f'{name[:30]:<30}' + ' '.join(f'{x:.3f}' for x in v))
