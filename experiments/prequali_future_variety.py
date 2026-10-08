"""How different are the pre-quali model's predictions from one upcoming race to the next?

Builds placeholder rows for every upcoming 2026 round (entry list + state as of the
latest completed race), lets data.add_features compute the walk-forward features,
trains the pre-quali model on 2018-2025 and predicts each round.

Between two upcoming races nothing new has happened, so only the track-dependent
inputs (TrackAvgShrunk, TrackType) can differ -- this measures how much that moves
the predicted order.
"""
import contextlib, io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'rebuild'))
import config
import data
from data import filter_races, split_years, fit_preprocessing, make_X
from model import build_model
import train

base = pd.concat([pd.read_csv(ROOT / 'data' / 'training_data.csv'), pd.read_csv(ROOT / 'data' / 'test_data.csv')],
                 ignore_index=True)
year = 2026
last = int(base.loc[base.Year == year, 'RoundNumber'].max())
latest = base[(base.Year == year) & (base.RoundNumber == last)].copy()
schedule = [s for s in json.loads((ROOT / 'data' / 'schedule_2026.json').read_text(encoding='utf-8'))
            if s['Year'] == year and s['RoundNumber'] > last]

# state AFTER the latest race
latest['SeasonPoints'] = latest['SeasonPoints'] + latest['Points'].fillna(0)
team_pts = latest.groupby('TeamName')['SeasonPoints'].sum().rank(ascending=False, method='min')
latest['ConstructorStanding'] = latest['TeamName'].map(team_pts)
track_type = base.sort_values(['Year', 'RoundNumber']).groupby('EventName')['TrackType'].last()

placeholders = []
for s in schedule:
    p = latest.copy()
    p['RoundNumber'], p['EventName'] = s['RoundNumber'], s['EventName']
    p['TrackType'] = track_type.get(s['EventName'], 0)
    p[['ActualPosition', 'ActualGridPosition', 'Points']] = np.nan
    p['IsDNF'] = False
    placeholders.append(p)
allrows = data.add_features(pd.concat([base] + placeholders, ignore_index=True))

grid = allrows['ActualGridPosition']
this_season = grid.groupby([allrows['DriverName'], allrows['Year']]).transform(lambda s: s.expanding().mean().shift(1))
per_season = grid.groupby([allrows['DriverName'], allrows['Year']]).mean()
allrows['ProjectedGrid'] = this_season.fillna(pd.Series(
    [per_season.get((d, y - 1), np.nan) for d, y in zip(allrows['DriverName'], allrows['Year'])], index=allrows.index))

pre = ['ProjectedGrid', 'PointsShare', 'TrackAvgShrunk', 'ConstructorStanding', 'RecentFormFin', 'TrackType']
config.FEATURE_COLS[:] = pre
train_df = split_years(filter_races(allrows[allrows['ActualPosition'].notna()]), config.TRAIN_YEARS)
torch.manual_seed(0)
med, sc = fit_preprocessing(train_df)
with contextlib.redirect_stdout(io.StringIO()):
    model, _ = train.train_model(build_model(len(pre), config.HIDDEN), make_X(train_df, med, sc),
                                 train_df['ActualPosition'].values, 40)
model.eval()

future = allrows[(allrows.Year == year) & (allrows.RoundNumber > last)]
orders = {}
for (r, ev), g in future.groupby(['RoundNumber', 'EventName']):
    with torch.no_grad():
        p = model(torch.FloatTensor(make_X(g, med, sc))).squeeze().numpy()
    orders[(r, ev)] = list(g.assign(p=p).sort_values('p')['DriverName'])

print(f'pre-quali model, upcoming 2026 rounds (state as of round {last})\n')
for (r, ev), o in orders.items():
    print(f'R{r:<3}{ev.replace(" Grand Prix", ""):<16}', ' '.join(o[:10]))
tops = [tuple(o[:10]) for o in orders.values()]
print(f'\ndistinct top-10 orders: {len(set(tops))} of {len(tops)} | distinct sets of top-10 drivers: {len({frozenset(t) for t in tops})}')
pos = pd.DataFrame({k[1].replace(' Grand Prix', ''): {d: i + 1 for i, d in enumerate(o)} for k, o in orders.items()})
spread = (pos.max(axis=1) - pos.min(axis=1)).sort_values(ascending=False)
print('biggest swings in predicted position across these races:', ', '.join(f'{d} {int(v)}' for d, v in spread.head(6).items()))
print('\nwhat differs between two upcoming races (per driver), Singapore vs Las Vegas:')
a, b = future[future.EventName.str.startswith('Singapore')].set_index('DriverName'), future[future.EventName.str.startswith('Las Vegas')].set_index('DriverName')
print({c: int((~np.isclose(a[c].astype(float), b.loc[a.index, c].astype(float), equal_nan=True)).sum()) for c in pre}, '(drivers whose value changes)')
