"""Does qualifying GAP-to-pole add information beyond the real grid slot?

Fetches qualifying times for 2018-2025 from Jolpica (Ergast mirror), then runs the
same leave-one-season-out CV used elsewhere, with a linear model (deterministic, so
no seed noise), scoring top-10 ranked MAE like rebuild/model.py:ranked_top10_mae.

Post-quali setting (the gap only exists after Saturday), so the base model uses the
REAL grid (ActualGridPosition) instead of the season-average grid.

Configs:  base | + QualiGapPct | + QualiPosition (pre-penalty) | + both
Baseline: rank by the real grid slot (what "just use qualifying order" scores).
"""
import json, sys, time, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'rebuild'))
from config import FEATURE_COLS, TRAIN_YEARS
from data import load_data, filter_races, split_years, split_train_val

API = 'https://api.jolpi.ca/ergast/f1'
CACHE = ROOT / 'data' / 'raw' / 'quali_jolpica.csv'


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'formula-forecast'}), timeout=30) as r:
                return json.load(r)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def secs(t):
    """'1:18.277' -> 78.277 ; '' / None -> NaN"""
    if not t:
        return np.nan
    try:
        if ':' not in t:                       # lap under a minute, or a bare number
            return float(t)
        m, s = t.split(':')
        return int(m) * 60 + float(s)
    except ValueError:                         # 'DNS' and similar
        return np.nan


def fetch_quali(years):
    rows = []
    for y in years:
        offset, total = 0, None
        while total is None or offset < total:
            j = get(f'{API}/{y}/qualifying.json?limit=100&offset={offset}')
            total = int(j['MRData']['total'])
            for race in j['MRData']['RaceTable']['Races']:
                for q in race['QualifyingResults']:
                    rows.append(dict(Year=y, RoundNumber=int(race['round']), RaceName=race['raceName'],
                                     DriverName=q['Driver']['code'], QualiPosition=int(q['position']),
                                     Q1=secs(q.get('Q1')), Q2=secs(q.get('Q2')), Q3=secs(q.get('Q3'))))
            offset += 100
            time.sleep(0.4)
        print(f'  {y}: {total} qualifying rows', flush=True)
    return pd.DataFrame(rows)


def ranked_mae(df, pred):
    d = df.copy(); d['p'] = pred
    d['r'] = d.groupby(['Year', 'RoundNumber'])['p'].rank(method='first')
    t = d[d['ActualPosition'] <= 10]
    return (t['r'] - t['ActualPosition']).abs().mean()


def main():
    if CACHE.exists():
        q = pd.read_csv(CACHE)
    else:
        print('fetching qualifying from Jolpica...')
        q = fetch_quali(range(2018, 2026))
        q.to_csv(CACHE, index=False)
    q['Best'] = q[['Q1', 'Q2', 'Q3']].min(axis=1)                       # fastest lap the driver set
    pole = q.groupby(['Year', 'RoundNumber'])['Best'].transform('min')
    q['QualiGapPct'] = (q['Best'] - pole) / pole * 100                  # % slower than pole
    q['QualiGapPct'] = q['QualiGapPct'].fillna(q.groupby(['Year', 'RoundNumber'])['QualiGapPct'].transform('max'))  # no time set -> worst gap in that session

    df = split_years(filter_races(load_data()), TRAIN_YEARS)
    m = df.merge(q[['Year', 'RoundNumber', 'DriverName', 'RaceName', 'QualiPosition', 'QualiGapPct']],
                 on=['Year', 'RoundNumber', 'DriverName'], how='left')
    print(f'\nmatched {m.QualiGapPct.notna().mean() * 100:.1f}% of {len(m)} rows')
    # do rounds line up? compare event names
    chk = m.dropna(subset=['RaceName']).drop_duplicates(['Year', 'RoundNumber'])
    same = (chk['EventName'].str.lower() == chk['RaceName'].str.lower()).mean()
    print(f'round alignment: event name identical for {same * 100:.0f}% of {len(chk)} races '
          f'(mismatches: {chk.loc[chk.EventName.str.lower() != chk.RaceName.str.lower(), ["Year", "RoundNumber", "EventName", "RaceName"]].head(5).values.tolist()})')
    m = m.dropna(subset=['QualiGapPct']).copy()

    post = [('ActualGridPosition' if c == 'GridPosition' else c) for c in FEATURE_COLS]
    configs = {'post-quali base': post,
               '+ QualiGapPct': post + ['QualiGapPct'],
               '+ QualiPosition (pre-penalty)': post + ['QualiPosition'],
               '+ both': post + ['QualiGapPct', 'QualiPosition']}
    res = {k: [] for k in configs}; grid = []
    for vy in TRAIN_YEARS:
        tr, va = split_train_val(m, vy)
        g = va.copy(); g['p'] = g['ActualGridPosition']
        grid.append(ranked_mae(va, va['ActualGridPosition'].values))
        for name, cols in configs.items():
            med = tr[cols].median()
            sc = StandardScaler().fit(tr[cols].fillna(med))
            lr = LinearRegression().fit(sc.transform(tr[cols].fillna(med)), tr['ActualPosition'])
            res[name].append(ranked_mae(va, lr.predict(sc.transform(va[cols].fillna(med)))))
    grid = np.array(grid)
    print(f'\nlinear model, 8-fold leave-one-season-out, top-10 ranked MAE (lower is better)')
    print(f'{"":34}{"CV mean":>8}{"vs grid":>9}{"folds beating grid":>20}')
    print(f'{"qualifying order (real grid slot)":34}{grid.mean():>8.3f}{0:>+9.3f}{"-":>20}')
    base = np.array(res['post-quali base'])
    for name, v in res.items():
        v = np.array(v)
        print(f'{name:34}{v.mean():>8.3f}{v.mean() - grid.mean():>+9.3f}{int((v < grid).sum()):>17}/8'
              + ('' if name == 'post-quali base' else f'   ({int((v < base).sum())}/8 folds better than base, {v.mean() - base.mean():+.3f})'))
    print('\nper fold:  ' + '  '.join(str(y) for y in TRAIN_YEARS))
    print(f'{"grid":<12}' + ' '.join(f'{x:.3f}' for x in grid))
    for name, v in res.items():
        print(f'{name[:12]:<12}' + ' '.join(f'{x:.3f}' for x in v))


if __name__ == '__main__':
    main()
