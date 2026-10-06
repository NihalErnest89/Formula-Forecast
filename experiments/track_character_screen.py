"""Track character (power / high-speed / twisty / balanced) + regularization screen.

Idea: tracks reward different car strengths, teams differ in which they are good at.
Feature TeamFormAtTrackType = the team's average finish THIS SEASON at tracks of the
same category, races before this one only (falls back to the team's season average,
then to NaN -> median fill). One-hot category inputs test whether the model can use
the category itself. Regularization configs use the reference features.

Same 8-fold leave-one-season-out CV and seeds as rebuild/train.py.
Labels below are a DRAFT for review; unsure: Spanish 2026 (Madrid), Mexico City,
Miami, Canadian.
"""
import contextlib, functools, io, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'rebuild'))
import config
from data import load_data, filter_races, split_years, split_train_val
import train

CATEGORY = {
    # power: long straights, low drag
    'Italian Grand Prix': 'power', 'Azerbaijan Grand Prix': 'power', 'Las Vegas Grand Prix': 'power',
    'Saudi Arabian Grand Prix': 'power', 'Belgian Grand Prix': 'power', 'Canadian Grand Prix': 'power',
    'Sakhir Grand Prix': 'power', 'Miami Grand Prix': 'power',
    # high-speed corners
    'British Grand Prix': 'highspeed', '70th Anniversary Grand Prix': 'highspeed', 'Japanese Grand Prix': 'highspeed',
    'Qatar Grand Prix': 'highspeed', 'Dutch Grand Prix': 'highspeed', 'Spanish Grand Prix': 'highspeed',
    'Barcelona Grand Prix': 'highspeed', 'Turkish Grand Prix': 'highspeed', 'Tuscan Grand Prix': 'highspeed',
    # twisty: slow corners, max downforce
    'Monaco Grand Prix': 'twisty', 'Hungarian Grand Prix': 'twisty', 'Singapore Grand Prix': 'twisty',
    # balanced
    'Australian Grand Prix': 'balanced', 'Bahrain Grand Prix': 'balanced', 'Chinese Grand Prix': 'balanced',
    'Austrian Grand Prix': 'balanced', 'Styrian Grand Prix': 'balanced', 'Abu Dhabi Grand Prix': 'balanced',
    'United States Grand Prix': 'balanced', 'Brazilian Grand Prix': 'balanced', 'São Paulo Grand Prix': 'balanced',
    'Emilia Romagna Grand Prix': 'balanced', 'French Grand Prix': 'balanced', 'German Grand Prix': 'balanced',
    'Portuguese Grand Prix': 'balanced', 'Russian Grand Prix': 'balanced', 'Eifel Grand Prix': 'balanced',
    'Mexican Grand Prix': 'balanced', 'Mexico City Grand Prix': 'balanced',
}
OVERRIDE = {('Spanish Grand Prix', 2026): 'balanced'}   # 2026 'Spanish GP' is Madrid, not Catalunya
CATS = ['power', 'highspeed', 'twisty', 'balanced']


def category(event, year):
    return OVERRIDE.get((event, year), CATEGORY.get(event))


raw = load_data()
raw['TrackCat'] = [category(e, y) for e, y in zip(raw['EventName'], raw['Year'])]
missing = sorted(set(raw.loc[raw['TrackCat'].isna(), 'EventName']))
assert not missing, f'unlabelled circuits: {missing}'

# per race, per team: average finishing position of its cars
team_race = (raw.groupby(['Year', 'RoundNumber', 'TeamName', 'TrackCat'], as_index=False)['ActualPosition']
             .mean().sort_values(['Year', 'RoundNumber']))
# walk-forward mean at the same category this season (races BEFORE this one)
g = team_race.groupby(['Year', 'TeamName', 'TrackCat'])['ActualPosition']
team_race['_cat_form'] = g.transform(lambda s: s.shift(1).expanding().mean())
# fallback: the team's season average so far, any category
g2 = team_race.groupby(['Year', 'TeamName'])['ActualPosition']
team_race['_season_form'] = g2.transform(lambda s: s.shift(1).expanding().mean())
team_race['TeamFormAtTrackType'] = team_race['_cat_form'].fillna(team_race['_season_form'])
print(f'TeamFormAtTrackType: {team_race["_cat_form"].notna().mean() * 100:.0f}% from same-category races, '
      f'{(team_race["_cat_form"].isna() & team_race["_season_form"].notna()).mean() * 100:.0f}% season fallback, '
      f'{team_race["TeamFormAtTrackType"].isna().mean() * 100:.0f}% empty (round 1 -> median)')

for c in CATS:
    raw['Cat_' + c] = (raw['TrackCat'] == c).astype(float)
raw = raw.merge(team_race[['Year', 'RoundNumber', 'TeamName', 'TeamFormAtTrackType']],
                on=['Year', 'RoundNumber', 'TeamName'], how='left')
new_cols = ['TeamFormAtTrackType'] + ['Cat_' + c for c in CATS]
keys = ['Year', 'RoundNumber', 'DriverName']
df = split_years(filter_races(load_data()), config.TRAIN_YEARS).merge(raw[keys + new_cols], on=keys, how='left')

# qualifying-order baseline on the same folds (rank by real grid)
grid = []
for vy in config.TRAIN_YEARS:
    _, va = split_train_val(df, vy)
    r = va.groupby(['Year', 'RoundNumber'])['ActualGridPosition'].rank(method='first')
    top = va['ActualPosition'] <= 10
    grid.append((r[top] - va.loc[top, 'ActualPosition']).abs().mean())
grid = np.array(grid)

base = list(config.FEATURE_COLS)
orig_adam, orig_build = torch.optim.Adam, train.build_model


def dropout_model(input_size, hidden, p):
    layers, prev = [], input_size
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


configs = {
    'reference': (base, {}),
    '+ TeamFormAtTrackType': (base + ['TeamFormAtTrackType'], {}),
    '+ track category (one-hot)': (base + ['Cat_' + c for c in CATS], {}),
    '+ both': (base + new_cols, {}),
    'weight decay 1e-4': (base, {'wd': 1e-4}),
    'weight decay 1e-3': (base, {'wd': 1e-3}),
    'dropout 0.2': (base, {'dropout': 0.2}),
}

results = {}
for name, (cols, reg) in configs.items():
    config.FEATURE_COLS[:] = cols
    torch.optim.Adam = functools.partial(orig_adam, weight_decay=reg['wd']) if 'wd' in reg else orig_adam
    train.build_model = functools.partial(dropout_model, p=reg['dropout']) if 'dropout' in reg else orig_build
    with contextlib.redirect_stdout(io.StringIO()):
        curves, importance = train.cross_validate(df)
    best = int(curves.mean(axis=0).argmin())
    results[name] = (curves[:, best], dict(zip(cols, importance)), best + 1)
    print(f'done: {name}', flush=True)
torch.optim.Adam, train.build_model = orig_adam, orig_build

ref = results['reference'][0]
print(f'\n{"":<30}{"CV mean":>8}{"vs ref":>8}{"better":>8}{"vs quali":>10}{"beats quali":>13}{"best ep":>9}')
print(f'{"qualifying order":<30}{grid.mean():>8.3f}{"":>16}{0:>+10.3f}')
for name, (v, imp, ep) in results.items():
    print(f'{name:<30}{v.mean():>8.3f}{v.mean() - ref.mean():>+8.3f}{int((v < ref).sum()):>6}/8'
          f'{v.mean() - grid.mean():>+10.3f}{int((v < grid).sum()):>11}/8{ep:>9}')
print('\nimportance of the new inputs:')
for name in ['+ TeamFormAtTrackType', '+ track category (one-hot)', '+ both']:
    imp = results[name][1]
    print(f'  {name:<28}', {k: round(float(x), 3) for k, x in imp.items() if k in new_cols or k == 'TrackType'})
print('\nper fold:  ' + '  '.join(str(y) for y in config.TRAIN_YEARS))
print(f'{"quali order":<26}' + ' '.join(f'{x:.3f}' for x in grid))
for name, (v, _, _) in results.items():
    print(f'{name[:26]:<26}' + ' '.join(f'{x:.3f}' for x in v))
