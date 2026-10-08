"""Do features rejected for the post-quali model help BEFORE qualifying?

Several were rejected because the real grid already carried their information
(qualifying at this track shows how the car suits it). The pre-quali model has no
real grid, so test them again, one at a time, on top of the pre-quali feature set.

Same 8-fold CV, seeds and training as rebuild/train.py; scored on the true-top-10
error and vs form order (rank by ProjectedGrid).
"""
import contextlib, io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'rebuild'))
sys.path.insert(0, str(ROOT / 'experiments'))
import config
import data
from data import load_data, filter_races, split_years, split_train_val
import train

RACE = ['Year', 'RoundNumber']
raw = load_data()                                         # sorted by Year, RoundNumber
fin = ~raw['IsDNF'].astype(bool) & raw['ActualPosition'].notna()
finish = raw['ActualPosition'].where(fin)

# ---- ProjectedGrid (same as the pre-quali screen / data.py step 1) ----------
grid = raw['ActualGridPosition']
this_season = grid.groupby([raw['DriverName'], raw['Year']]).transform(lambda s: s.expanding().mean().shift(1))
per_season = grid.groupby([raw['DriverName'], raw['Year']]).mean()
raw['ProjectedGrid'] = this_season.fillna(pd.Series(
    [per_season.get((d, y - 1), np.nan) for d, y in zip(raw['DriverName'], raw['Year'])], index=raw.index))

# ---- track category + team form at this track type (track_character_screen) --
import ast  # read ONLY the label dicts from track_character_screen.py (importing it would rerun that test)
_tree = ast.parse((ROOT / 'experiments' / 'track_character_screen.py').read_text(encoding='utf-8'))
_labels = {t.id: ast.literal_eval(n.value) for n in _tree.body if isinstance(n, ast.Assign)
           for t in n.targets if isinstance(t, ast.Name) and t.id in ('CATEGORY', 'OVERRIDE', 'CATS')}
CATEGORY, OVERRIDE, CATS = _labels['CATEGORY'], _labels['OVERRIDE'], _labels['CATS']
raw['TrackCat'] = [OVERRIDE.get((e, y), CATEGORY.get(e)) for e, y in zip(raw['EventName'], raw['Year'])]
team_race = (raw.groupby(RACE + ['TeamName', 'TrackCat'], as_index=False)['ActualPosition'].mean()
             .sort_values(RACE))
cat_form = team_race.groupby(['Year', 'TeamName', 'TrackCat'])['ActualPosition'].transform(lambda s: s.shift(1).expanding().mean())
season_form = team_race.groupby(['Year', 'TeamName'])['ActualPosition'].transform(lambda s: s.shift(1).expanding().mean())
team_race['TeamFormAtTrackType'] = cat_form.fillna(season_form)
raw = raw.merge(team_race[RACE + ['TeamName', 'TeamFormAtTrackType']], on=RACE + ['TeamName'], how='left')
for c in CATS:
    raw['Cat_' + c] = (raw['TrackCat'] == c).astype(float)

# ---- overtaking difficulty per circuit (overtaking_screen) -------------------
circ = pd.Series([data.circuit_of(e, y) for e, y in zip(raw['EventName'], raw['Year'])], index=raw.index)
races = []
for (y, r), g in raw[fin].groupby(RACE):
    races.append(dict(Year=y, RoundNumber=r, C=circ[g.index[0]],
                      rho=g['ActualGridPosition'].corr(g['ActualPosition'], method='spearman')))
races = pd.DataFrame(races).sort_values(RACE)
races['GridStickiness'] = races.groupby('C')['rho'].transform(lambda s: s.shift(1).expanding().mean()).fillna(
    races['rho'].shift(1).expanding().mean())
raw = raw.merge(races[RACE + ['GridStickiness']], on=RACE, how='left')

# ---- track affinity: better/worse than usual here (feature_cleanup_screen) ---
prior = lambda s: s.cumsum().shift(1)
tot, cnt = finish.fillna(0), finish.notna().astype(float)
overall = tot.groupby(raw['DriverName']).transform(prior) / cnt.groupby(raw['DriverName']).transform(prior).replace(0, np.nan)
raw['TrackAffinity'] = raw['TrackAvgShrunk'] - overall

# ---- wins (wins_window_screen) ------------------------------------------------
win = (raw['ActualPosition'] == 1).astype(int)
raw['WinsThisSeason'] = win.groupby([raw['DriverName'], raw['Year']]).transform(lambda s: s.shift(1).fillna(0).cumsum())

new = ['ProjectedGrid', 'TeamFormAtTrackType', 'GridStickiness', 'TrackAffinity', 'WinsThisSeason'] + ['Cat_' + c for c in CATS]
keys = RACE + ['DriverName']
df = split_years(filter_races(load_data()), config.TRAIN_YEARS)
df = df.drop(columns=[c for c in new if c in df.columns]).merge(raw[keys + new], on=keys, how='left')

form = []
for vy in config.TRAIN_YEARS:
    _, va = split_train_val(df, vy)
    rk = va.groupby(RACE)['ProjectedGrid'].rank(method='first'); t = va['ActualPosition'] <= 10
    form.append((rk[t] - va.loc[t, 'ActualPosition']).abs().mean())
form = np.array(form)

base = ['ProjectedGrid', 'PointsShare', 'TrackAvgShrunk', 'ConstructorStanding', 'RecentFormFin', 'TrackType']
configs = {
    'pre-quali reference': base,
    '+ TeamFormAtTrackType': base + ['TeamFormAtTrackType'],
    '+ track category (one-hot)': base + ['Cat_' + c for c in CATS],
    '+ GridStickiness (overtaking)': base + ['GridStickiness'],
    '+ TrackAffinity': base + ['TrackAffinity'],
    '+ ConstructorTrackAvg': base + ['ConstructorTrackAvg'],
    '+ HistoricalTrackAvgPosition (old)': base + ['HistoricalTrackAvgPosition'],
    '+ CareerWins': base + ['CareerWins'],
    '+ WinsLast3Years': base + ['WinsLast3Years'],
    '+ WinsThisSeason': base + ['WinsThisSeason'],
    'combined track: TeamForm + HistTrackAvg + category': base + ['TeamFormAtTrackType', 'HistoricalTrackAvgPosition'] + ['Cat_' + c for c in CATS],
    'combined track: TeamForm + HistTrackAvg': base + ['TeamFormAtTrackType', 'HistoricalTrackAvgPosition'],
}
only = sys.argv[1:]
if only:
    configs = {k: v for k, v in configs.items() if any(k.startswith(o) for o in only)}
results = {}
for name, cols in configs.items():
    config.FEATURE_COLS[:] = cols
    with contextlib.redirect_stdout(io.StringIO()):
        curves, imp = train.cross_validate(df)
    v = curves[:, int(curves.mean(0).argmin())]
    added = [c for c in cols if c not in base]
    results[name] = (v, {c: float(x) for c, x in zip(cols, imp) if c in added})
    print(f'done: {name}', flush=True)

ref = results['pre-quali reference'][0]
print(f'\nform order (projected grid): {form.mean():.3f}\n')
print(f'{"":<36}{"CV mean":>8}{"vs ref":>8}{"better":>8}{"vs form":>9}   importance of the addition')
for name, (v, imp) in results.items():
    print(f'{name:<36}{v.mean():>8.3f}{v.mean() - ref.mean():>+8.3f}{int((v < ref).sum()):>6}/8{v.mean() - form.mean():>+9.3f}   '
          + ', '.join(f'{k} {x:.3f}' for k, x in imp.items()))
print('\nper fold:  ' + '  '.join(str(y) for y in config.TRAIN_YEARS))
for name, (v, _) in results.items():
    print(f'{name[:30]:<30}' + ' '.join(f'{x:.3f}' for x in v))
