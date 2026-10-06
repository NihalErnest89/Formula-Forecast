"""Adam + L2 (weight_decay) vs AdamW (decoupled weight decay), same CV as rebuild/train.py.

Each config replaces BOTH torch.optim.Adam and torch.optim.AdamW with the chosen optimizer +
weight decay, so it works whichever one train.py calls.
"""
import contextlib, io, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'rebuild'))
import config
from data import load_data, filter_races, split_years
import train

df = split_years(filter_races(load_data()), config.TRAIN_YEARS)
ADAM, ADAMW = torch.optim.Adam, torch.optim.AdamW
configs = {
    'Adam, no weight decay': (ADAM, 0.0),
    'Adam, weight_decay 1e-4': (ADAM, 1e-4),
    'Adam, weight_decay 1e-3 (current)': (ADAM, 1e-3),
    'AdamW, default (0.01)': (ADAMW, 0.01),
    'AdamW, 0.1': (ADAMW, 0.1),
}
results = {}
def use(opt, wd):
    # replace whichever optimizer train.py calls, ignoring the weight_decay it passes (if any)
    factory = lambda params, lr, **_: opt(params, lr=lr, weight_decay=wd)
    torch.optim.Adam = torch.optim.AdamW = factory


for name, (opt, wd) in configs.items():
    use(opt, wd)
    with contextlib.redirect_stdout(io.StringIO()):
        curves, _ = train.cross_validate(df)
    best = int(curves.mean(axis=0).argmin())
    results[name] = curves[:, best]
    print(f'done: {name}', flush=True)
torch.optim.Adam, torch.optim.AdamW = ADAM, ADAMW

ref = results['Adam, no weight decay']
print(f'\n{"":<36}{"CV mean":>8}{"vs none":>9}{"seasons better":>16}')
for name, v in results.items():
    print(f'{name:<36}{v.mean():>8.3f}{v.mean() - ref.mean():>+9.3f}{int((v < ref).sum()):>13}/8')
print('\nper fold:  ' + '  '.join(str(y) for y in config.TRAIN_YEARS))
for name, v in results.items():
    print(f'{name[:34]:<34}' + ' '.join(f'{x:.3f}' for x in v))
