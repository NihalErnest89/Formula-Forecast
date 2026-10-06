"""Clean up the existing features (instead of adding new ones), real-grid model.

Each config changes ONE thing vs the current 6 features. Same 8-fold leave-one-season-out
CV and seeds as rebuild/train.py (whatever optimizer train.py currently uses).

Points (SeasonPoints grows 0 -> ~500 through a season, so the same number means
different things at round 3 and round 20):
  PointsPerRace     = points / races run so far            (keeps gaps, no drift)
  PointsShare       = points / the leader's points (0-1)   (keeps gaps as proportions)
  SeasonStanding    = rank 1-22                             (no drift, loses gaps)
Recent form (season-scoped, falls back to last season's value at the start of a year):
  RecentFormFin     = mean finish over the last 5 races the driver FINISHED
  RecentGain        = mean places gained (grid - finish) over the last 5 finishes
Track history (renamed circuits merged; finishes only; shrunk toward the driver's
overall average when they have few races there, k = 2 pseudo-races):
  TrackAvgShrunk    = shrunk average finish at this circuit
  TrackAffinity     = TrackAvgShrunk - driver's overall average (better/worse than usual here)
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

SAME_CIRCUIT = {'70th Anniversary Grand Prix': 'British Grand Prix', 'Styrian Grand Prix': 'Austrian Grand Prix',
                'Mexican Grand Prix': 'Mexico City Grand Prix', 'Brazilian Grand Prix': 'São Paulo Grand Prix',
                'Barcelona Grand Prix': 'Spanish Grand Prix'}
K_SHRINK = 2.0

raw = load_data().sort_values(['Year', 'RoundNumber']).reset_index(drop=True)
raw.loc[raw['RoundNumber'] == 1, 'SeasonPoints'] = 0          # same as filter_races
fin = ~raw['IsDNF'].astype(bool) & raw['ActualPosition'].notna()
race = ['Year', 'RoundNumber']

# ---- points --------------------------------------------------------------
raw['PointsPerRace'] = raw['SeasonPoints'] / (raw['RoundNumber'] - 1).clip(lower=1)
leader = raw.groupby(race)['SeasonPoints'].transform('max')
raw['PointsShare'] = (raw['SeasonPoints'] / leader.replace(0, np.nan)).fillna(0)


# ---- recent form (finishes only), season-scoped with last-season fallback ----
def last_k_of(values, keys, k=5):
    """mean of the last k non-NaN values BEFORE each row, within each group."""
    def f(s):
        out = s.dropna().rolling(k, min_periods=1).mean()
        return out.reindex(s.index).ffill().shift(1)
    return values.groupby([raw[c] for c in keys], group_keys=False).apply(f)


pos_fin = raw['ActualPosition'].where(fin)
gain_fin = (raw['ActualGridPosition'] - raw['ActualPosition']).where(fin)
for name, vals in [('RecentFormFin', pos_fin), ('RecentGain', gain_fin)]:
    season = last_k_of(vals, ['DriverName', 'Year'])
    carry = last_k_of(vals, ['DriverName'])                        # crosses seasons: used only as fallback
    raw[name] = season.fillna(carry)

# ---- track history, shrunk ------------------------------------------------
raw['_circuit'] = [('Madrid' if e == 'Spanish Grand Prix' and y >= 2026 else SAME_CIRCUIT.get(e, e))
                   for e, y in zip(raw['EventName'], raw['Year'])]
pos0 = pos_fin.fillna(0); cnt = pos_fin.notna().astype(float)
drv_sum = pos0.groupby(raw['DriverName']).transform(lambda s: s.cumsum().shift(1))
drv_n = cnt.groupby(raw['DriverName']).transform(lambda s: s.cumsum().shift(1))
trk_sum = pos0.groupby([raw['DriverName'], raw['_circuit']]).transform(lambda s: s.cumsum().shift(1)).fillna(0)
trk_n = cnt.groupby([raw['DriverName'], raw['_circuit']]).transform(lambda s: s.cumsum().shift(1)).fillna(0)
overall = drv_sum / drv_n.replace(0, np.nan)                       # NaN for a debut -> median fill
raw['TrackAvgShrunk'] = (trk_sum + K_SHRINK * overall) / (trk_n + K_SHRINK)
raw['TrackAffinity'] = raw['TrackAvgShrunk'] - overall

new = ['PointsPerRace', 'PointsShare', 'RecentFormFin', 'RecentGain', 'TrackAvgShrunk', 'TrackAffinity']
keys = race + ['DriverName']
df = split_years(filter_races(load_data()), config.TRAIN_YEARS)
df = df.drop(columns=[c for c in new if c in df.columns]).merge(raw[keys + new], on=keys, how='left')   # data.py may already add some
print('missing after merge (-> median fill):', {c: f'{df[c].isna().mean() * 100:.1f}%' for c in new}, flush=True)

grid = []
for vy in config.TRAIN_YEARS:
    _, va = split_train_val(df, vy)
    rk = va.groupby(race)['ActualGridPosition'].rank(method='first')
    top = va['ActualPosition'] <= 10
    grid.append((rk[top] - va.loc[top, 'ActualPosition']).abs().mean())
grid = np.array(grid)

base = list(config.FEATURE_COLS)
swap = lambda old, new_: [new_ if c == old else c for c in base]
configs = {
    'reference': base,
    'SeasonPoints -> PointsPerRace': swap('SeasonPoints', 'PointsPerRace'),
    'SeasonPoints -> PointsShare': swap('SeasonPoints', 'PointsShare'),
    'SeasonPoints -> SeasonStanding': swap('SeasonPoints', 'SeasonStanding'),
    'RecentForm -> RecentFormFin': swap('RecentForm', 'RecentFormFin'),
    'RecentForm -> RecentGain': swap('RecentForm', 'RecentGain'),
    '+ RecentGain': base + ['RecentGain'],
    'HistTrackAvg -> TrackAvgShrunk': swap('HistoricalTrackAvgPosition', 'TrackAvgShrunk'),
    'HistTrackAvg -> TrackAffinity': swap('HistoricalTrackAvgPosition', 'TrackAffinity'),
    'combined: PointsShare + RecentFormFin + TrackAvgShrunk': [
        {'SeasonPoints': 'PointsShare', 'RecentForm': 'RecentFormFin',
         'HistoricalTrackAvgPosition': 'TrackAvgShrunk'}.get(c, c) for c in base],
}
only = sys.argv[1:]
if only:
    configs = {k: v for k, v in configs.items() if any(k.startswith(o) for o in only)}
results = {}
for name, cols in configs.items():
    config.FEATURE_COLS[:] = cols
    with contextlib.redirect_stdout(io.StringIO()):
        curves, importance = train.cross_validate(df)
    best = int(curves.mean(axis=0).argmin())
    results[name] = (curves[:, best], dict(zip(cols, importance)))
    print(f'done: {name}', flush=True)

ref = results['reference'][0]
print(f'\n{"":<34}{"CV mean":>8}{"vs ref":>8}{"better":>8}{"vs quali":>10}   importance of changed input')
print(f'{"qualifying order":<34}{grid.mean():>8.3f}')
for name, (v, imp) in results.items():
    changed = [c for c in imp if c not in base]
    print(f'{name:<34}{v.mean():>8.3f}{v.mean() - ref.mean():>+8.3f}{int((v < ref).sum()):>6}/8{v.mean() - grid.mean():>+10.3f}   '
          + ', '.join(f'{c} {imp[c]:.3f}' for c in changed))
print('\nreference importance:', {k: round(float(x), 3) for k, x in results['reference'][1].items()})
print('\nper fold:  ' + '  '.join(str(y) for y in config.TRAIN_YEARS))
print(f'{"quali order":<30}' + ' '.join(f'{x:.3f}' for x in grid))
for name, (v, _) in results.items():
    print(f'{name[:30]:<30}' + ' '.join(f'{x:.3f}' for x in v))
