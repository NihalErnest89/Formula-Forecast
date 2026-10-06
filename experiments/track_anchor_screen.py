"""TrackAvgShrunk: what should the "usual result" anchor be?

TrackAvgShrunk = (sum of finishes at this track + k * anchor) / (finishes at this track + k)
The anchor in rebuild/data.py is the driver's CAREER average (all races since 2018), which
mixes in old cars (Hamilton's Mercedes years in 2026). Alternatives:
  season  = average finish THIS season so far (career average until they've finished a race)
  last20  = average of the driver's last 20 finishes (about one season, crosses the winter)
Base features = the combined clean-up set (1.788). Same 8-fold CV and seeds as rebuild/train.py.
"""
import contextlib, io, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'rebuild'))
import config
import data
from data import load_data, filter_races, split_years
import train

raw = load_data()     # already sorted by Year, RoundNumber; has TrackAvgShrunk (career anchor)
fin = ~raw['IsDNF'].astype(bool) & raw['ActualPosition'].notna()
finish = raw['ActualPosition'].where(fin)
prior = lambda s: s.cumsum().shift(1)
total, count = finish.fillna(0), finish.notna().astype(float)
circuit = pd.Series([data.circuit_of(e, y) for e, y in zip(raw['EventName'], raw['Year'])], index=raw.index)

career = (total.groupby(raw['DriverName']).transform(prior)
          / count.groupby(raw['DriverName']).transform(prior).replace(0, np.nan))
season = (total.groupby([raw['DriverName'], raw['Year']]).transform(prior)
          / count.groupby([raw['DriverName'], raw['Year']]).transform(prior).replace(0, np.nan)).fillna(career)
last20 = finish.groupby(raw['DriverName'], group_keys=False).apply(
    lambda s: s.dropna().rolling(20, min_periods=1).mean().reindex(s.index).ffill().shift(1))

trk_total = total.groupby([raw['DriverName'], circuit]).transform(prior).fillna(0)
trk_count = count.groupby([raw['DriverName'], circuit]).transform(prior).fillna(0)
for name, anchor in [('career', career), ('season', season), ('last20', last20)]:
    raw['TAS_' + name] = (trk_total + data.SHRINK * anchor) / (trk_count + data.SHRINK)

same = np.allclose(raw['TAS_career'].fillna(-1), raw['TrackAvgShrunk'].fillna(-1))
print(f'career version matches data.py TrackAvgShrunk: {same}', flush=True)

keys = ['Year', 'RoundNumber', 'DriverName']
cols = ['TAS_career', 'TAS_season', 'TAS_last20']
df = split_years(filter_races(load_data()), config.TRAIN_YEARS).merge(raw[keys + cols], on=keys, how='left')

base = ['ActualGridPosition', 'PointsShare', 'TrackAvgShrunk', 'ConstructorStanding', 'RecentFormFin', 'TrackType']
configs = {f'anchor = {c[4:]}': [c if f == 'TrackAvgShrunk' else f for f in base] for c in cols}
results = {}
for name, feats in configs.items():
    config.FEATURE_COLS[:] = feats
    with contextlib.redirect_stdout(io.StringIO()):
        curves, imp = train.cross_validate(df)
    v = curves[:, int(curves.mean(0).argmin())]
    results[name] = (v, imp[feats.index([f for f in feats if f.startswith('TAS_')][0])])
    print(f'done: {name}', flush=True)

ref = results['anchor = career'][0]
print(f'\n{"":<20}{"CV mean":>8}{"vs career":>11}{"seasons better":>16}{"importance":>12}')
for name, (v, imp) in results.items():
    print(f'{name:<20}{v.mean():>8.3f}{v.mean() - ref.mean():>+11.3f}{int((v < ref).sum()):>13}/8{imp:>12.3f}')
print('\nper fold:  ' + '  '.join(str(y) for y in config.TRAIN_YEARS))
for name, (v, _) in results.items():
    print(f'{name:<20}' + ' '.join(f'{x:.3f}' for x in v))
