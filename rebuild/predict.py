# practice for my own knowledge of the process we followed

import os
import sys

import torch
from config import DATA_DIR, DEFAULT_YEAR, MODELS, TEST_YEARS
from data import filter_races, load_data, make_X, next_race, split_years
from model import evaluate, load_artifacts, permutation_importance, weight_shares
from train import main as train_main
import pandas as pd




# ---------------------------------------------------------------------------
# 10. predict
# ---------------------------------------------------------------------------
# scale with the LOADED scaler (not a new one), predict, then rank within the
# race. no groupby needed here -- it's a single race, so the group is implicit.

def predict_race(model, scaler, medians, race_df, features):
    race = race_df.copy()
    X = make_X(race, medians, scaler, features)

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


def ask_race(test_data, next_df=None):
    """List this season's races, each labelled with the model it uses, and ask for
    a round. Returns (race rows, year, round, model name)."""
    year = DEFAULT_YEAR
    done = test_data[test_data['Year'] == year]
    print(f'\n{year} races:')
    for rnd, event in done[['RoundNumber', 'EventName']].drop_duplicates().itertuples(index=False):
        print(f'  R{int(rnd):<3} {event:<28} [qualifying known  -> post-quali model]')
    next_round, next_model = None, None
    if next_df is not None and len(next_df):
        next_round = int(next_df['RoundNumber'].iloc[0])
        if next_df['ActualGridPosition'].notna().all():      # qualifying saved by collect_data.py
            next_model, label = 'postquali', '[qualifying known  -> post-quali model, race not run yet]'
        else:
            next_model, label = 'prequali', '[no qualifying yet -> pre-quali model, a projection]'
        print(f'  R{next_round:<3} {next_df["EventName"].iloc[0]:<28} {label}')

    while True:
        try:
            rnd = int(input('  round: ').strip())
        except ValueError:
            print('  numbers only, try again')
            continue

        race = done[done['RoundNumber'] == rnd]
        if len(race):
            return race, year, rnd, 'postquali'
        if rnd == next_round:
            return next_df, year, rnd, next_model
        print(f'  no race found for {year} round {rnd}')


# ---------------------------------------------------------------------------
# evaluate both models
# ---------------------------------------------------------------------------
# same layout as train.py's summary: each model's scores on the test season,
# then importance and weight tables with one column per model.

def evaluate_both(test_df):
    loaded, importance, weights = {}, {}, {}
    for name, spec in MODELS.items():
        model, scaler, medians = load_artifacts(spec)
        features = spec['features']
        X = make_X(test_df, medians, scaler, features)

        print(f'\n=============== {name} model on {TEST_YEARS} ===============')
        evaluate(model, test_df, X, spec['grid'])

        grid_label = {spec['grid']: 'grid (real / projected)'}
        importance[name] = pd.Series(permutation_importance(model, test_df, X), index=features).rename(grid_label)
        weights[name] = weight_shares(model, features).rename(grid_label)
        loaded[name] = (model, scaler, medians)

    for title, per_model in [('feature importance', importance), ('first-layer weight share', weights)]:
        table = pd.DataFrame(per_model)
        print(f'\n{title}:')
        print(table.sort_values(table.columns[0], ascending=False).round(3).to_string())
    return loaded


# Update races
def collect_races():
    repo_root = DATA_DIR.parent
    previous_dir = os.getcwd()
    os.chdir(repo_root)
    try:
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import collect_data
        collect_data.main()
    except Exception as e:
        print(f'data update failed ({e}), keeping the data already on disk')
    finally:
        os.chdir(previous_dir)

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    saved = all((s['dir'] / 'model.pth').exists() for s in MODELS.values())
    if not saved:
        print('no saved model -- run python train.py first')
        train_main()
    elif '--train' in sys.argv or ask_yes_no('saved model found. retrain it?'):
        train_main()


    upcoming = next_race(DEFAULT_YEAR)          # (year, round, event) or None
    print('loading data...')
    full = load_data(upcoming)
    df = filter_races(full)                     # drops the next race's rows (no result yet)
    test_df = split_years(df, TEST_YEARS)
    next_df = None
    if upcoming:
        next_df = full[(full['Year'] == upcoming[0]) & (full['RoundNumber'] == upcoming[1])]
    print(f'  {len(test_df)} test rows ({TEST_YEARS})')

    loaded = evaluate_both(test_df)

    
    choice = input('1. Predict a race\n2. Update races:\n')
    if choice == '1':
        while True:
            race, year, rnd, name = ask_race(test_df, next_df)
            model, scaler, medians = loaded[name]
            features = MODELS[name]['features']
            print(f"\npredicting {year} round {rnd} ({race['EventName'].iloc[0]}) with the {name} model...")
            result = predict_race(model, scaler, medians, race, features)
            cols = ['DriverName', 'pred', 'pred_rank', 'ActualPosition'] + features
            print(result[cols].round(2).to_string(index=False))
            if not ask_yes_no('\npredict another race?', default=True):
                break
    elif choice == '2':
        collect_races()
    else:
        print(f"Invalid choice")
        



if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# later (don't do this yet)
# ---------------------------------------------------------------------------
# once the above works end to end, the delta idea: instead of predicting
# finish position, predict (finish - grid) and add it to the grid rank.
# that only makes sense after you've seen how close naive quali order
# already gets -- that comparison is the whole motivation for it.
