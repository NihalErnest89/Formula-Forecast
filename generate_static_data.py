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

from top10.model_loader import load_model
from top10.predict import predict_race_top10
from top10.feature_calculation import calculate_future_race_features
from top10.data_utils import load_f1_data, build_race_list, format_predictions

OUT_DIR = Path(__file__).parent / 'frontend' / 'public' / 'data'
DATA_DIR = Path(__file__).parent / 'data'
MODEL_DIR = Path(__file__).parent / 'models'


def generate_race_prediction(race, test_df, training_df, model, scaler, model_type, device):
    year = int(race['year'])
    event_name = race['eventName']
    round_number = int(race['roundNumber'])
    is_future = race.get('isFuture', False)

    race_df = test_df[
        (test_df['Year'] == year) &
        (test_df['EventName'] == event_name) &
        (test_df['RoundNumber'] == round_number)
    ].copy()

    if race_df.empty:
        race_df = calculate_future_race_features(
            test_df, year, round_number, event_name, training_df
        )
        is_future = True

    if race_df.empty:
        print(f"  Skipping {event_name} ({year}) — no driver data available")
        return None

    top10, all_drivers = predict_race_top10(race_df, model, scaler, model_type, device)
    predictions_filtered, predictions_unfiltered = format_predictions(top10, is_future)

    return {
        'race': {
            'year': year,
            'eventName': event_name,
            'roundNumber': round_number,
            'isFuture': is_future,
        },
        'predictionsFiltered': predictions_filtered,
        'predictionsUnfiltered': predictions_unfiltered,
        'totalDrivers': len(all_drivers),
    }


def main():
    print("Loading model...")
    model, scaler, model_type, device = load_model(str(MODEL_DIR), 'neural_network')
    print(f"  Model loaded ({model_type}, device={device})")

    print("Loading data...")
    test_df, training_df = load_f1_data(DATA_DIR)
    if test_df is None:
        print("ERROR: test_data.csv not found. Run data collection first.")
        sys.exit(1)

    races_list = build_race_list(test_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_dir = OUT_DIR / 'predictions'
    predictions_dir.mkdir(exist_ok=True)

    races_path = OUT_DIR / 'races.json'
    with open(races_path, 'w') as f:
        json.dump({'races': races_list}, f)
    print(f"Wrote {races_path} ({len(races_list)} races)")

    ok = 0
    skipped = 0
    for race in races_list:
        future_tag = ' [FUTURE]' if race['isFuture'] else ''
        print(f"  Predicting: {race['eventName']} ({race['year']}, Round {race['roundNumber']}){future_tag}...")
        try:
            data = generate_race_prediction(race, test_df, training_df, model, scaler, model_type, device)
            if data is None:
                skipped += 1
                continue
            out_path = predictions_dir / f"{race['year']}-{race['roundNumber']}.json"
            with open(out_path, 'w') as f:
                json.dump(data, f)
            ok += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            skipped += 1

    print(f"\nDone. {ok} predictions written, {skipped} skipped.")
    print(f"Output: {OUT_DIR}")


if __name__ == '__main__':
    main()
