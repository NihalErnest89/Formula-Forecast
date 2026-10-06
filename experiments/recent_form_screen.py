"""How should 'recent form' be defined, and which weak features can go?

Same 8-fold leave-one-season-out CV, seeds and early stopping as rebuild/train.py
(the reference row should reproduce the reference score).

RecentForm today = mean finishing position over the last 5 races. Two suspects:
  * window: 5 races may be too short (noisy) or too long (stale)
  * DNFs (IsDNF now flags all ~590 retirements, not just 22): a retirement is recorded as a finish around P18-22, so ONE crash adds ~3
    places to a 5-race average even though it says nothing about pace.

Variants are recomputed walk-forward from the results (only races BEFORE each one,
across season boundaries, keyed by driver identity). "last 5 (recomputed)" is a control:
it should land close to the reference.
"""
import contextlib, io, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'rebuild'))
import config
from data import load_data, filter_races, split_years
import train

raw = load_data().sort_values(['Year', 'RoundNumber']).reset_index(drop=True)
raw['_pos'] = raw['ActualPosition']
raw['_dnf'] = raw['IsDNF'].fillna(False).astype(bool)


def per_driver(fn):
    """fn(group_df) -> Series aligned to the group's index; stitched back together."""
    return raw.groupby('DriverName', group_keys=False).apply(fn)


def last_k(k):
    return per_driver(lambda g: g['_pos'].shift(1).rolling(k, min_periods=1).mean())


def ewm(halflife):
    return per_driver(lambda g: g['_pos'].shift(1).ewm(halflife=halflife, ignore_na=True).mean())


def last_k_finishers(k):
    """mean of the last k races the driver actually FINISHED (retirements ignored)."""
    def f(g):
        fin = g['_pos'].where(~g['_dnf'])                     # NaN on a retirement
        out = fin.dropna().rolling(k, min_periods=1).mean()   # over finishes only
        return out.reindex(g.index).ffill().shift(1)          # carry forward, exclude today
    return per_driver(f)


variants = {
    'last 5 (recomputed, control)': last_k(5),
    'last 3': last_k(3),
    'last 8': last_k(8),
    'last 12': last_k(12),
    'EWM halflife 3 races': ewm(3),
    'last 5 FINISHES (DNFs ignored)': last_k_finishers(5),
    'last 8 FINISHES (DNFs ignored)': last_k_finishers(8),
}
for name, s in variants.items():
    raw['RF::' + name] = s.values if hasattr(s, 'values') else s

df = split_years(filter_races(load_data()), config.TRAIN_YEARS)
keys = ['Year', 'RoundNumber', 'DriverName']
df = df.merge(raw[keys + ['RF::' + n for n in variants]], on=keys, how='left')

base = list(config.FEATURE_COLS)
swap = lambda new, old='RecentForm': [new if c == old else c for c in base]
drop = lambda *names: [c for c in base if c not in names]
configs = {'reference': base}
configs.update({f'RecentForm -> {n}': swap('RF::' + n) for n in variants})
configs['drop CareerWins'] = drop('CareerWins')
configs['drop ConstructorTrackAvg'] = drop('ConstructorTrackAvg')
configs['drop CareerWins + ConstructorTrackAvg'] = drop('CareerWins', 'ConstructorTrackAvg')

results = {}
for name, cols in configs.items():
    config.FEATURE_COLS[:] = cols
    with contextlib.redirect_stdout(io.StringIO()):
        curves, _ = train.cross_validate(df)
    best = int(curves.mean(axis=0).argmin())
    results[name] = (curves[:, best], best + 1)
    print(f'done: {name}', flush=True)

ref = results['reference'][0]
print(f'\n{"":<64}{"CV mean":>8}{"vs ref":>8}{"folds better":>14}')
for name, (v, ep) in results.items():
    print(f'{name:<64}{v.mean():>8.3f}{v.mean() - ref.mean():>+8.3f}{int((v < ref).sum()):>11}/8')
print('\nper fold:  ' + '  '.join(str(y) for y in config.TRAIN_YEARS))
for name, (v, _) in results.items():
    print(f'{name[:30]:<30}' + ' '.join(f'{x:.3f}' for x in v))
