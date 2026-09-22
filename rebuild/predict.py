# practice for my own knowledge of the process we followed

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from config import *
from data import filter_races, load_data, make_X, prepare_features, split_train_val
from model import build_model, evaluate, load_artifacts, save_artifacts, train_model

DATA_DIR = Path(__file__).parent.parent / 'data'
OUT = Path(__file__).parent / 'saved'






# ---------------------------------------------------------------------------
# 10. predict
# ---------------------------------------------------------------------------
# scale with the LOADED scaler (not a new one), predict, then rank within the
# race. no groupby needed here -- it's a single race, so the group is implicit.

def predict_race(model, scaler, medians, race_df):
    race = race_df.copy()
    X = make_X(race, medians, scaler)

    model.eval()
    with torch.no_grad():
        race['pred'] = model(torch.FloatTensor(X)).squeeze().numpy()

    race['pred_rank'] = race['pred'].rank(method='first')
    return race.sort_values('pred_rank')


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
# input() always hands back a STRING, so anything compared against a dataframe
# column has to be int()-ed first -- otherwise "2026" != 2026 and every lookup
# silently finds nothing.
#
# the isatty() checks let the script still run unattended (piped, or from
# another script) instead of blowing up on EOF when there's nobody to type.

def ask_yes_no(question, default=False):
    if not sys.stdin.isatty():
        return default
    suffix = '[Y/n]' if default else '[y/N]'
    answer = input(f'{question} {suffix}: ').strip().lower()
    if not answer:
        return default
    return answer.startswith('y')


def ask_race(test_data):
    """Prompt for year + round, re-asking until it names a race we actually have."""
    available = test_data.groupby('Year')['RoundNumber'].agg(['min', 'max'])
    print('\navailable races:')
    for year, row in available.iterrows():
        print(f'  {year}: rounds {int(row["min"])}-{int(row["max"])}')

    if not sys.stdin.isatty():
        year = int(available.index[-1])
        rnd = int(available.iloc[-1]['max'])
        print(f'  (not a terminal, defaulting to {year} round {rnd})')
        return test_data[(test_data['Year'] == year) & (test_data['RoundNumber'] == rnd)], year, rnd

    while True:
        try:
            year = int(input('  year: ').strip())
            rnd = int(input('  round: ').strip())
        except ValueError:
            print('  numbers only, try again')
            continue

        race = test_data[(test_data['Year'] == year) & (test_data['RoundNumber'] == rnd)]
        if race.empty:
            print(f'  no race found for {year} round {rnd}')
            continue
        return race, year, rnd


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    force_train = '--train' in sys.argv
    have_saved = (OUT / 'model.pth').exists()

    print('loading data...')
    training_data, test_data = load_data()
    training_data = filter_races(training_data)
    test_data = filter_races(test_data)
    print(f'  {len(training_data)} training rows, {len(test_data)} test rows')

    # decide whether to train: no saved model leaves no choice, --train forces
    # it, otherwise ask.
    if not have_saved:
        print('\nno saved model found, training a new one')
        retrain = True
    elif force_train:
        print('\n--train given, retraining')
        retrain = True
    else:
        retrain = ask_yes_no('\nsaved model found. retrain it?')

    if retrain:
        train_data, val_data = split_train_val(training_data, VAL_YEAR)
        prep = prepare_features(train_data, val_data, test_data)

        print(f'\ntraining on {len(prep["X_train"])} rows, validating on {len(prep["X_val"])} ({VAL_YEAR})')
        model = build_model(len(FEATURE_COLS), HIDDEN)
        model, _ = train_model(model, prep['X_train'], prep['y_train'],
                               prep['X_val'], prep['y_val'])

        print('\nsaving...')
        save_artifacts(model, prep['scaler'], prep['medians'])
        scaler, medians = prep['scaler'], prep['medians']
    else:
        print('\nusing the saved model (pass --train to retrain)')
        model, scaler, medians = load_artifacts()

    # evaluate -- note X_test is rebuilt here from the scaler/medians we ended
    # up with, whether those came from training or from disk. same path either way.
    print('\nevaluating on 2025-2026...')
    evaluate(model, test_data, make_X(test_data, medians, scaler))

    # predict races until told to stop
    while True:
        race, year, rnd = ask_race(test_data)
        print(f'\npredicting {year} round {rnd}...')
        result = predict_race(model, scaler, medians, race)
        print(result[['DriverName', 'ActualGridPosition', 'pred', 'pred_rank', 'ActualPosition']].to_string(index=False))

        if not ask_yes_no('\npredict another race?'):
            break


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# later (don't do this yet)
# ---------------------------------------------------------------------------
# once the above works end to end, the delta idea: instead of predicting
# finish position, predict (finish - grid) and add it to the grid rank.
# that only makes sense after you've seen how close naive quali order
# already gets -- that comparison is the whole motivation for it.
