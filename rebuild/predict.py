# practice for my own knowledge of the process we followed

import sys

import torch
from config import OUT, TEST_YEARS
from data import filter_races, load_data, make_X, split_years
from model import evaluate, load_artifacts
from train import main as train_main





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
    saved = (OUT / 'model.pth').exists()
    if not saved:
        print('no saved model -- run python train.py first')
        train_main()
    elif '--train' in sys.argv or ask_yes_no('saved model found. retrain it?'):
        train_main()

    model, scaler, medians = load_artifacts()

    print('loading data...')
    df = filter_races(load_data())
    test_df = split_years(df, TEST_YEARS)
    print(f'  {len(test_df)} test rows ({TEST_YEARS})')

    print(f'\nevaluating on {TEST_YEARS}...')
    evaluate(model, test_df, make_X(test_df, medians, scaler))

    while True:
        race, year, rnd = ask_race(test_df)
        print(f'\npredicting {year} round {rnd}...')
        result = predict_race(model, scaler, medians, race)
        print(result[['DriverName', 'GridPosition', 'pred', 'pred_rank', 'ActualPosition']].to_string(index=False))

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
