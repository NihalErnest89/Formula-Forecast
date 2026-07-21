"""
Generate static JSON data files for the GitHub Pages frontend.

Run this after training to pre-compute all predictions:
    python generate_static_data.py

Outputs:
    frontend/public/data/races.json
    frontend/public/data/predictions/<year>-<roundNumber>.json
"""

import json
import sys
from pathlib import Path

# Add repo root to path so we can import top10 modules
sys.path.insert(0, str(Path(__file__).parent))

from top10.model_loader import load_model, load_postquali_model, load_prequali_delta_model
from top10.predict import predict_race_top10, predict_race_postquali
from top10.feature_calculation import (calculate_future_race_features, add_racecraft_features,
                                       add_overqual_features, add_elo_features)
from top10.data_utils import load_f1_data, build_race_list, format_predictions

OUT_DIR = Path(__file__).parent / 'frontend' / 'public' / 'data'
DATA_DIR = Path(__file__).parent / 'data'
MODEL_DIR = Path(__file__).parent / 'models'
CACHE_DIR = Path(__file__).parent / 'cache'


def _canon_driver(x):
    """Normalize a driver number so 4, 4.0, '4' all map to '4'."""
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return str(x)


def try_fetch_quali_grid(year, event_name):
    """
    For an upcoming race, check whether qualifying has already happened and
    return {driver_number(str): grid_position} if so, else None.

    Only meaningful for the NEXT race of a season (later rounds cannot have
    quali results yet). Failures (no session, API down, not run yet) return None.
    """
    try:
        import fastf1
        import pandas as pd
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        session = fastf1.get_session(year, event_name, 'Q')
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        results = session.results
        if results is None or results.empty or results['Position'].isna().all():
            return None
        return {
            _canon_driver(dn): float(pos)
            for dn, pos in zip(results['DriverNumber'], results['Position'])
            if pd.notna(pos)
        }
    except Exception:
        return None


def generate_race_prediction(race, test_df, training_df, model, scaler, model_type, device,
                             pq_model=None, pq_scaler=None, pq_meta=None,
                             pre_model=None, pre_scaler=None, pre_meta=None,
                             latest_driver_gain=None, latest_team_gain=None,
                             latest_driver_elo=None, latest_team_elo=None):
    year = int(race['year'])
    event_name = race['eventName']
    round_number = int(race['roundNumber'])
    is_future = race.get('isFuture', False)

    race_df = test_df[
        (test_df['Year'] == year) &
        (test_df['EventName'] == event_name) &
        (test_df['RoundNumber'] == round_number)
    ].copy()

    used_quali_grid = False
    model_used = 'legacy'

    def attach_latest(df):
        """Attach current racecraft + Elo features (as of latest completed race)."""
        df['DriverAvgGain'] = df['DriverNumber'].map(
            lambda d: (latest_driver_gain or {}).get(_canon_driver(d), 0.0))
        df['DriverElo'] = df['DriverNumber'].map(
            lambda d: (latest_driver_elo or {}).get(_canon_driver(d), 1500.0))
        if 'TeamName' in df.columns:
            df['ConstructorAvgGain'] = df['TeamName'].map(
                lambda t: (latest_team_gain or {}).get(str(t), 0.0))
            df['ConstructorElo'] = df['TeamName'].map(
                lambda t: (latest_team_elo or {}).get(str(t), 1500.0))
        else:
            df['ConstructorAvgGain'] = 0.0
            df['ConstructorElo'] = 1500.0

    if race_df.empty:
        race_df = calculate_future_race_features(
            test_df, year, round_number, event_name, training_df
        )
        is_future = True
        if race_df.empty:
            print(f"  Skipping {event_name} ({year}) — no driver data available")
            return None
        # Upcoming race: if this is the season's next round, qualifying may have
        # already happened - if so, switch to actual grid + post-quali model
        quali_grid = None
        if pq_model is not None and race.get('isNextRound'):
            quali_grid = try_fetch_quali_grid(year, event_name)
        if quali_grid:
            mapped = race_df['DriverNumber'].map(lambda d: quali_grid.get(_canon_driver(d)))
            if mapped.notna().any():
                # Season-average grid (the projected value) BEFORE swapping in quali grid
                race_df['SeasonAvgGrid'] = race_df['GridPosition']
                race_df['GridPosition'] = mapped.fillna(race_df['GridPosition'])
                attach_latest(race_df)
                add_overqual_features(race_df)
                used_quali_grid = True
                model_used = 'postquali-delta'
                print(f"    Using ACTUAL qualifying grid (quali completed) + post-quali delta model")
        if not used_quali_grid and pre_model is not None:
            # Future race without quali: pre-quali delta ensemble over form order
            attach_latest(race_df)
            model_used = 'prequali-delta'
    else:
        # Completed race: qualifying is known, so use the actual grid with the
        # post-quali model (quali precedes the race - no information leakage)
        if pq_model is not None and 'ActualGridPosition' in race_df.columns \
                and race_df['ActualGridPosition'].notna().mean() > 0.5:
            race_df['SeasonAvgGrid'] = race_df['GridPosition']
            race_df['GridPosition'] = race_df['ActualGridPosition'].fillna(race_df['GridPosition'])
            add_overqual_features(race_df)
            used_quali_grid = True
            model_used = 'postquali-delta'

    if used_quali_grid:
        top10, all_drivers = predict_race_postquali(race_df, pq_model, pq_scaler, pq_meta, device)
    elif model_used == 'prequali-delta':
        top10, all_drivers = predict_race_postquali(race_df, pre_model, pre_scaler, pre_meta, device)
    else:
        top10, all_drivers = predict_race_top10(race_df, model, scaler, model_type, device)
    predictions_filtered, predictions_unfiltered = format_predictions(top10, is_future)

    return {
        'race': {
            'year': year,
            'eventName': event_name,
            'roundNumber': round_number,
            'isFuture': is_future,
            'usedQualiGrid': used_quali_grid,
            'modelUsed': model_used,
        },
        'predictionsFiltered': predictions_filtered,
        'predictionsUnfiltered': predictions_unfiltered,
        'totalDrivers': len(all_drivers),
    }


# F1 points system (race only; future sprint points are not modeled)
POINTS_BY_RANK = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def generate_standings(year, test_df, predictions):
    """
    Build championship standings for a season.

    Two accumulations from the per-race prediction JSONs:
    - predictedPoints: model points for FUTURE races -> projectedTotal/Rank
      (current points + predicted remainder = projected championship)
    - modelPoints: model points for COMPLETED races -> modelRank
      ("the championship the model predicted" - for finished seasons this is
      compared against the actual final standings)

    Args:
        year: season year
        test_df: DataFrame of completed races (test_data.csv)
        predictions: list of this year's prediction dicts (future AND completed)

    Returns:
        dict ready to serialize as standings/<year>.json
    """
    season = test_df[test_df['Year'] == year].copy()
    if season.empty:
        return None
    season['_dn'] = season['DriverNumber'].apply(_canon_driver)

    # Actual points per driver (Points column = race + sprint, source of truth)
    drivers = {}
    for dn, grp in season.groupby('_dn'):
        latest = grp.sort_values('RoundNumber').iloc[-1]
        drivers[dn] = {
            'driverNumber': dn,
            'driverName': latest.get('DriverName', 'UNK'),
            'teamName': latest.get('TeamName', ''),
            'currentPoints': float(grp['Points'].fillna(0).sum()),
            'predictedPoints': 0.0,
            'modelPoints': 0.0,
        }

    def _driver_entry(p):
        dn = _canon_driver(p.get('driverNumber'))
        if dn not in drivers:
            drivers[dn] = {
                'driverNumber': dn,
                'driverName': p.get('driverName', 'UNK'),
                'teamName': p.get('constructor') or '',
                'currentPoints': 0.0,
                'predictedPoints': 0.0,
                'modelPoints': 0.0,
            }
        return drivers[dn]

    n_future = 0
    for pred in predictions:
        is_future = pred['race']['isFuture']
        n_future += int(is_future)
        for p in pred['predictionsFiltered']:
            pts = POINTS_BY_RANK.get(p.get('rank'))
            if pts is None:
                continue
            entry = _driver_entry(p)
            if is_future:
                entry['predictedPoints'] += pts   # projection of the remainder
            else:
                entry['modelPoints'] += pts       # what the model predicted pre-race

    rows = list(drivers.values())
    for r in rows:
        r['projectedTotal'] = r['currentPoints'] + r['predictedPoints']

    # Ranks: current (actual points so far), projected (with predicted future
    # races), and model (championship built purely from per-race predictions)
    for key, rank_field in (('currentPoints', 'currentRank'),
                            ('projectedTotal', 'projectedRank'),
                            ('modelPoints', 'modelRank')):
        for i, r in enumerate(sorted(rows, key=lambda x: -x[key]), 1):
            r[rank_field] = i

    rows.sort(key=lambda r: -r['projectedTotal'])

    completed_rounds = sorted(int(x) for x in season['RoundNumber'].unique())
    return {
        'year': int(year),
        'completedRounds': len(completed_rounds),
        'remainingRounds': n_future,
        'seasonComplete': n_future == 0,
        'note': 'Projected totals = actual points from completed races + model-predicted points '
                '(25-18-15-...-1) for each remaining race. Model points = the same points system '
                'applied to the model\'s pre-race predictions of completed races. '
                'Sprint points are not modeled.',
        'standings': rows,
    }


def main():
    print("Loading model...")
    model, scaler, model_type, device = load_model(str(MODEL_DIR), 'neural_network')
    print(f"  Model loaded ({model_type}, device={device})")

    pq_model, pq_scaler, pq_meta = load_postquali_model(str(MODEL_DIR))
    if pq_model is None:
        print("  No post-quali model found - completed races will use the average-grid model.")
        print("  (Retrain with 'python top10/train.py' to enable post-quali predictions.)")
    pre_model, pre_scaler, pre_meta = load_prequali_delta_model(str(MODEL_DIR))
    if pre_model is None:
        print("  No pre-quali delta model found - future races will use the legacy model.")

    print("Loading data...")
    test_df, training_df = load_f1_data(DATA_DIR)
    if test_df is None:
        print("ERROR: test_data.csv not found. Run data collection first.")
        sys.exit(1)

    # Walk-forward racecraft + Elo features (also yields CURRENT values for future races)
    latest_driver_gain, latest_team_gain = {}, {}
    latest_driver_elo, latest_team_elo = {}, {}
    if pq_model is not None or pre_model is not None:
        _, latest_driver_gain, latest_team_gain = add_racecraft_features([training_df, test_df])
        _, latest_driver_elo, latest_team_elo = add_elo_features([training_df, test_df])

    races_list = build_race_list(test_df)

    # Mark each season's next upcoming round: it is the only future race whose
    # qualifying could already have results worth probing for
    next_rounds = {}
    for r in races_list:
        if r.get('isFuture'):
            yr = r['year']
            if yr not in next_rounds or r['roundNumber'] < next_rounds[yr]:
                next_rounds[yr] = r['roundNumber']
    for r in races_list:
        r['isNextRound'] = bool(r.get('isFuture')) and next_rounds.get(r['year']) == r['roundNumber']

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_dir = OUT_DIR / 'predictions'
    predictions_dir.mkdir(exist_ok=True)

    races_path = OUT_DIR / 'races.json'
    with open(races_path, 'w') as f:
        json.dump({'races': races_list}, f)
    print(f"Wrote {races_path} ({len(races_list)} races)")

    ok = 0
    skipped = 0
    preds_by_year = {}
    for race in races_list:
        future_tag = ' [FUTURE]' if race['isFuture'] else ''
        print(f"  Predicting: {race['eventName']} ({race['year']}, Round {race['roundNumber']}){future_tag}...")
        try:
            data = generate_race_prediction(race, test_df, training_df, model, scaler, model_type, device,
                                            pq_model=pq_model, pq_scaler=pq_scaler, pq_meta=pq_meta,
                                            pre_model=pre_model, pre_scaler=pre_scaler, pre_meta=pre_meta,
                                            latest_driver_gain=latest_driver_gain,
                                            latest_team_gain=latest_team_gain,
                                            latest_driver_elo=latest_driver_elo,
                                            latest_team_elo=latest_team_elo)
            if data is None:
                skipped += 1
                continue
            out_path = predictions_dir / f"{race['year']}-{race['roundNumber']}.json"
            with open(out_path, 'w') as f:
                json.dump(data, f)
            ok += 1
            preds_by_year.setdefault(race['year'], []).append(data)
        except Exception as e:
            print(f"    ERROR: {e}")
            skipped += 1

    # Projected championship standings per season
    standings_dir = OUT_DIR / 'standings'
    standings_dir.mkdir(exist_ok=True)
    years = sorted({r['year'] for r in races_list})
    for year in years:
        standings = generate_standings(year, test_df, preds_by_year.get(year, []))
        if standings is None:
            continue
        with open(standings_dir / f'{year}.json', 'w') as f:
            json.dump(standings, f)
        leader = standings['standings'][0]
        print(f"  Standings {year}: {standings['completedRounds']} done + {standings['remainingRounds']} predicted; "
              f"projected champion: {leader['driverName']} ({leader['projectedTotal']:.0f} pts)")

    print(f"\nDone. {ok} predictions written, {skipped} skipped.")
    print(f"Output: {OUT_DIR}")


if __name__ == '__main__':
    main()
